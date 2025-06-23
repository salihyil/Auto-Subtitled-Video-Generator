import streamlit as st
from streamlit_lottie import st_lottie
import mlx.core as mx
import mlx_whisper
import requests
from typing import List, Dict, Any
import pathlib
import os
import base64
import logging
from zipfile import ZipFile
import subprocess
import numpy as np
import re
import yt_dlp
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEVICE = "mps" if mx.metal.is_available() else "cpu"
MODELS = {
    "Tiny (Q4)": "mlx-community/whisper-tiny-mlx-q4",
    "Large v3": "mlx-community/whisper-large-v3-mlx",
    "Small English (Q4)": "mlx-community/whisper-small.en-mlx-q4",
    "Small (FP32)": "mlx-community/whisper-small-mlx-fp32",
    "Distil Large v3 (English)": "mlx-community/distil-whisper-large-v3",
    "Large v3 Turbo": "mlx-community/whisper-large-v3-turbo"
}
APP_DIR = pathlib.Path(__file__).parent.absolute()
LOCAL_DIR = APP_DIR / "local_video"
LOCAL_DIR.mkdir(exist_ok=True)
SAVE_DIR = LOCAL_DIR / "output"
SAVE_DIR.mkdir(exist_ok=True)
LANGUAGES = {
    "Detect automatically": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Hebrew": "he",
    "Turkish": "tr"
}

# Utility functions
@st.cache_data
def load_lottie_url(url: str) -> Dict[str, Any]:
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logging.error(f"Failed to load Lottie animation: {e}")
        return None

def filter_turkish_text(text: str) -> str:
    """Filter and correct common Turkish transcription errors."""
    # Replace common mistranscriptions
    replacements = {
        # Türkçe özel karakterler için düzeltmeler
        'sh': 'ş',
        'ch': 'ç',
        'gh': 'ğ',
        # Yaygın hatalı tanımalar için düzeltmeler
        'turkce': 'türkçe',
        'turkiye': 'türkiye'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def download_youtube_video(url: str) -> str:
    """YouTube videosunu indirip geçici bir dosya olarak kaydet."""
    try:
        logging.info(f"Downloading YouTube video: {url}")
        
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "video.mp4")
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        logging.info(f"YouTube video downloaded to: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"YouTube indirme hatası: {e}")
        raise

def prepare_audio(audio_path: str) -> mx.array:
    command = [
        "ffmpeg", "-i", audio_path, "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_data, _ = process.communicate()
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return mx.array(audio_array)

def process_audio(model_path: str, audio: mx.array, task: str, language: str = None) -> Dict[str, Any]:
    logging.info(f"Processing audio with model: {model_path}, task: {task}, language: {language}")
    try:
        decode_options = {"language": language} if language else {}

        if task.lower() == "transcribe":
            results = mlx_whisper.transcribe(
                audio, path_or_hf_repo=model_path, fp16=False, verbose=True, word_timestamps=True, **decode_options
            )
            logging.info(f"{task.capitalize()} completed successfully")
            return results
        else:
            raise ValueError(f"Unsupported task: {task}")
    except Exception as e:
        logging.error(f"Unexpected error in mlx_whisper.{task}: {e}")
        raise

def format_timestamp(seconds: float, vtt: bool = False) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if vtt:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    else:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace('.', ',')

def create_download_link(file_path: str, link_text: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

# Subtitle and transcription functions
def split_text_into_lines(text: str, max_chars: int = 42) -> List[str]:
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_chars:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def write_subtitles(segments: List[Dict[str, Any]], format: str, output_file: str, remove_fillers: bool = True) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        if format == "vtt":
            f.write("WEBVTT\n\n")
        
        subtitle_count = 1
        for segment in segments:
            words = segment.get('words', [])
            if not words:
                continue
            
            text = ' '.join(word['word'] for word in words)
            if remove_fillers:
                text = re.sub(r'\b(um|uh)\b', '', text).strip()
            
            lines = split_text_into_lines(text)
            
            for i in range(0, len(lines), 2):
                subtitle_lines = lines[i:i+2]
                subtitle_text = '\n'.join(subtitle_lines)
                
                start_index = sum(len(line.split()) for line in lines[:i])
                end_index = min(sum(len(line.split()) for line in lines[:i+2]), len(words))
                
                start_word = words[start_index]
                end_word = words[end_index - 1]
                
                start_time = start_word['start']
                end_time = end_word['end']
                
                duration = end_time - start_time
                min_duration = max(len(subtitle_text) / 21, 1.5)
                if duration < min_duration:
                    end_time = start_time + min_duration
                
                if format == "srt":
                    f.write(f"{subtitle_count}\n")
                    f.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                    f.write(f"{subtitle_text}\n\n")
                elif format == "vtt":
                    f.write(f"{format_timestamp(start_time, vtt=True)} --> {format_timestamp(end_time, vtt=True)}\n")
                    f.write(f"{subtitle_text}\n\n")
                
                subtitle_count += 1
            
            check_data_loss(segment, lines)

    check_final_output(segments, output_file)

def write_text_transcription(segments: List[Dict[str, Any]], output_file: str, remove_fillers: bool = True) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            text = segment['text']
            if remove_fillers:
                text = re.sub(r'\b(um|uh)\b', '', text).strip()
            f.write(text + "\n")

def check_data_loss(segment: Dict[str, Any], processed_lines: List[str]) -> None:
    processed_words = ' '.join(processed_lines).split()
    original_words = ' '.join(word['word'] for word in segment['words']).split()
    if len(processed_words) != len(original_words):
        logging.warning(f"Potential data loss detected in segment {segment.get('id', 'unknown')}")
        logging.warning(f"Original: {' '.join(original_words)}")
        logging.warning(f"Processed: {' '.join(processed_words)}")

def check_final_output(segments: List[Dict[str, Any]], output_file: str) -> None:
    original_text = ' '.join(seg['text'] for seg in segments)
    final_text = ' '.join(line.strip() for line in open(output_file, 'r', encoding='utf-8').readlines() if line.strip() and not line[0].isdigit() and '-->' not in line)
    if original_text != final_text:
        logging.warning("Potential data loss or word order change detected in final output")

# Streamlit UI functions
def render_header():
    col1, col2 = st.columns([1, 3])
    with col1:
        lottie = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_HjK9Ol.json")
        if lottie:
            st_lottie(lottie)
    with col2:
        st.markdown("""
            ## Apple MLX Powered Video Transcription

            Yükle veya YouTube URL'si girerek:
            - Doğru transkriptler (SRT/VTT dosyaları)
            - Hızlı işleme

            🎙️ Transkripsiyon: Orijinal dilde konuşulanları yazıya dökün
        """)

def render_model_selection():
    selected_model = st.selectbox("Whisper Modeli Seçin", list(MODELS.keys()), index=4)
    if selected_model == "Distil Large v3 (English)":
        st.info("""
        **Distil Large v3 Model**
        
        Bu yeni model önemli performans iyileştirmeleri sunar:
        - M1 Max çiplerde gerçek zamanlı işlemeden yaklaşık 40 kat daha hızlı çalışır
        - 12 dakikalık ses dosyasını sadece 18 saniyede transkribe edebilir
        - Hız ve doğruluk arasında harika bir denge sağlar
        
        Daha uzun videoları işlemek veya doğruluktan fazla ödün vermeden hızlı sonuçlar almak istediğinizde idealdir.
        """)
    if selected_model == "Large v3 Turbo":
        st.info("""
        **Large v3 Turbo**
        
        Bu yeni model önemli performans iyileştirmeleri sunar:
        - M2 Ultra'da 12 dakikayı 14 saniyede transkribe eder (gerçek zamandan ~50X daha hızlı)
        - Large v3 modeline göre önemli ölçüde daha küçüktür (809M vs 1550M)
        - Çok dilli destekler
        """)
    if selected_model in ["Small English (Q4)", "Distil Large v3 (English)"]:
        return MODELS[selected_model], True
    else:
        return MODELS[selected_model], False

def process_video(input_file, model_name, language, is_youtube=False):
    try:
        if is_youtube:
            input_path = input_file  # YouTube için indirilen video yolunu kullan
        else:
            input_path = str(SAVE_DIR / "input.mp4")
            with open(input_path, "wb") as f:
                f.write(input_file.read())
        
        audio = prepare_audio(input_path)
        results = process_audio(model_name, audio, "transcribe", language)
        
        # Türkçe dil düzeltmesi uygula
        is_turkish = language == "tr"
        if is_turkish:
            # Metni düzelt
            results["text"] = filter_turkish_text(results["text"])
            # Segmentleri düzelt
            for segment in results["segments"]:
                segment["text"] = filter_turkish_text(segment["text"])
                # Kelimeleri düzelt (varsa)
                if "words" in segment:
                    for word in segment["words"]:
                        if "word" in word:
                            word["word"] = filter_turkish_text(word["word"])
        
        vtt_path = str(SAVE_DIR / "transcript.vtt")
        srt_path = str(SAVE_DIR / "transcript.srt")
        txt_path = str(SAVE_DIR / "transcript.txt")
        
        write_subtitles(results["segments"], "vtt", vtt_path)
        write_subtitles(results["segments"], "srt", srt_path)
        write_text_transcription(results["segments"], txt_path)
        
        zip_path = str(SAVE_DIR / "transcripts.zip")
        with ZipFile(zip_path, "w") as zipf:
            for file in [vtt_path, srt_path, txt_path]:
                zipf.write(file, os.path.basename(file))
        
        st.markdown(create_download_link(zip_path, "Transkriptleri İndir"), unsafe_allow_html=True)
        
        if is_youtube:
            st.video(input_path)
        else:
            st.video(input_file)
            
        # Dil seçimine göre başlık değiştir
        title = "Transkripsiyon" if language == "tr" else "Transcription"
        st.text_area(title, results["text"], height=300)
        
        success_message = "Transkripsiyon başarıyla tamamlandı!" if language == "tr" else f"Transcription completed successfully using {model_name} model!"
        st.success(success_message)
    
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        logging.exception("Error in processing media")

def main():
    st.set_page_config(page_title="Auto Subtitled Video Generator", page_icon=":movie_camera:", layout="wide")
    render_header()
    
    # Sekmeleri oluştur
    tab1, tab2 = st.tabs(["Dosya Yükle", "YouTube URL"])
    
    # Model ve dil seçimi
    model_name, is_language_locked = render_model_selection()
   
    if is_language_locked:
        selected_language = "English"
        st.selectbox("Dil seçin", ["English"], disabled=True)
    else:
        selected_language = st.selectbox("Dil seçin", list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index("Turkish") if "Turkish" in LANGUAGES else 0)
    
    if selected_language == "Turkish":
        st.info("""
        **Türkçe Dil Desteği**
        
        Bu model Türkçe konuşmaları doğru şekilde algılayıp transkribe edebilir.
        - Türkçeye özgü karakterler (ç, ş, ğ, ü, ö, ı) doğru şekilde tanınır
        - Türkçe tonlama ve aksan özelliklerine uyumludur
        - En iyi sonuç için "Large v3" veya "Large v3 Turbo" modellerini kullanınız
        """)
    
    language = LANGUAGES[selected_language]

    with tab1:
        input_file = st.file_uploader("Medya Dosyası Yükle", type=["mp4", "avi", "mov", "mkv", "wav", "mp3", "opus", "m4a", "ogg"])
        if input_file and st.button("Transkribe Et", key="upload_button"):
            with st.spinner(f"{model_name} modeli kullanılarak medya transkribe ediliyor..."):
                process_video(input_file, model_name, language)
    
    with tab2:
        youtube_url = st.text_input("YouTube URL'si Girin")
        if youtube_url and st.button("YouTube'dan Transkribe Et", key="youtube_button"):
            with st.spinner("YouTube videosu indiriliyor ve transkribe ediliyor..."):
                try:
                    video_path = download_youtube_video(youtube_url)
                    process_video(video_path, model_name, language, is_youtube=True)
                except Exception as e:
                    st.error(f"YouTube video indirme veya işleme hatası: {str(e)}")

if __name__ == "__main__":
    main()
