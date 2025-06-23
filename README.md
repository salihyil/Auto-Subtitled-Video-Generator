# Apple MLX Powered Video Transcription

This Streamlit application allows users to upload video files or enter YouTube URLs and generate accurate transcripts using Apple's MLX framework.

## Important Note

⚠️ This application is designed to run on Apple Silicon (M series) Macs only. It utilizes the MLX framework, which is optimized for Apple's custom chips.

## Getting Started

### Prerequisites

- Apple Silicon (M series) Mac
- Conda package manager

If you don't have Conda installed on your Mac, you can follow the [Ultimate Guide to Installing Miniforge for AI Development on M1 Macs](https://www.rayfernando.ai/ultimate-guide-installing-miniforge-ai-development-m1-macs) for a comprehensive setup process.

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/salihyil/Auto-Subtitled-Video-Generator.git;
   cd MLX-Auto-Subtitled-Video-Generator
   ```

2. Create a new Conda environment with Python 3.12:

   ```
   conda create -n mlx-whisper python=3.12;
   conda activate mlx-whisper
   ```

3. Install the required dependencies:

   ```
   xcode-select --install
   pip install -r requirements.txt
   ```

4. Install FFmpeg (required for audio processing):

   ```
   brew install ffmpeg
   ```

   Note: If you don't have Homebrew installed, you can install it by running the following command in your terminal:

   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   After installation, follow the instructions provided in the terminal to add Homebrew to your PATH. For more information about Homebrew, visit [brew.sh](https://brew.sh/).

### Running the Application

To run the Streamlit application, use the following command:

```
streamlit run mlx_whisper_transcribe.py
```

## Features

- Upload video files (MP4, AVI, MOV, MKV)
- Direct video transcription using YouTube URLs
- Transcribe videos using various Whisper models
- Generate VTT and SRT subtitle files
- Download transcripts as a ZIP file
- Multi-language support including Turkish

## How It Works

1. Upload a video file or enter a YouTube URL
2. Choose a Whisper model
3. Click the "Transcribe" button to process the video
4. View the results and download the generated transcripts

### Using with YouTube URLs

1. Go to the "YouTube URL" tab
2. Enter the URL of the YouTube video you want to transcribe
3. Click "Transcribe from YouTube" button
4. The video will be automatically downloaded and transcribed
5. Once the process is complete, you can download the transcript files

## Models

The application supports the following Whisper models:

- Tiny (Q4)
- Large v3
- Small English (Q4)
- Small (FP32)
- Distil Large v3
- Large v3 Turbo (New!)

Each model has different capabilities and processing speeds. Experiment with different models to find the best balance between accuracy and performance for your needs.

### New Model: Large v3 Turbo

The newly added Large v3 Turbo model offers significant performance improvements:

- Transcribes 12 minutes in 14 seconds on an M2 Ultra (~50X faster than real time)
- Significantly smaller than the Large v3 model (809M vs 1550M)
- It is multilingual

This model is particularly useful for processing longer videos or when you need quick results without sacrificing too much accuracy.

## Troubleshooting

If you encounter any issues, please check the following:

- Ensure you're using an Apple Silicon Mac
- Verify that all dependencies are correctly installed
- Check the console output for any error messages

For any persistent problems, please open an issue in the repository.

## Acknowledgements

This project is a fork of the [original Auto-Subtitled Video Generator](https://github.com/BatuhanYilmaz26/Auto-Subtitled-Video-Generator) by Batuhan Yilmaz. I deeply appreciate the contribution to the open-source community.

Additionally, this project is based on RayFernando1337's MLX-Auto-Subtitled-Video-Generator repository. I would like to thank him for his contributions to the development of this application.
