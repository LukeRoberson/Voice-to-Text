# Video to Subtitle Generator

A Python application that automatically generates subtitles from MP4 video files using OpenAI's Whisper AI model with GPU acceleration via the faster-whisper implementation.

</br></br>


## Features

- 🚀 **GPU-Accelerated**: Optimized for NVIDIA GPUs with CUDA support
- 🎯 **High Accuracy**: Uses state-of-the-art Whisper AI model
- 📝 **Multiple Formats**: Supports SRT and WebVTT subtitle formats
- 🔧 **Flexible Model Selection**: Choose from multiple model sizes (tiny to large)
- 🎬 **Easy to Use**: Simple command-line interface
- 🔊 **Automatic Audio Extraction**: Handles video-to-audio conversion internally

</br></br>


## Requirements

### System Requirements

- Python
  - 3.8 or higher
  - Virtual environment (optional)
- NVIDIA GPU (or CPU as fallback)
  - Install latest NVIDIA drivers
  - CUDA (12.8 tested); https://developer.nvidia.com/cuda-12-8-0-download-archive
- FFmpeg (for audio extraction)
  - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Save locally, or add to PATH

</br></br>


### Python Dependencies
- faster-whisper
  - OpenAI's voice model
  - `pip install faster-whisper`
- torch and torchaudio (PyTorch with CUDA support)
  - Optional - For better GPU support
  - https://pytorch.org/get-started/locally/
  - Try to get a version that matches your CUDA version
  - For example: `pip3 install torch --index-url https://download.pytorch.org/whl/cu128`

> [!NOTE]
> Pytorch is only used for GPU detection

</br></br>


### Validate Environment

Run the validation script to confirm everything is ready to go.

```bash
python validate_setup.py
```

</br></br>


## Usage

### Basic Usage

Generate subtitles from a video file (uses default 'base' model):

```powershell
python main.py path\to\your\video.mp4
```

</br></br>


### Advanced Options

**Specify model size** (larger = more accurate but slower):
```powershell
python main.py video.mp4 -m medium
```
</br></br>


**Change output format to WebVTT**:
```powershell
python main.py video.mp4 -f vtt
```
</br></br>


**Specify output directory**:
```powershell
python main.py video.mp4 -o .\subtitles
```
</br></br>


**Combine options**:
```powershell
python main.py video.mp4 -m large-v3 -f vtt -o .\output
```
</br></br>


### Command-Line Arguments

| Argument       | Short | Description                        | Default       |
|----------------|-------|------------------------------------|---------------|
| `video_path`   | -     | Path to input MP4 video (required) | -             |
| `--output-dir` | `-o`  | Output directory for subtitle file | Same as video |
| `--model-size` | `-m`  | Whisper model size                 | `base`        |
| `--format`     | `-f`  | Subtitle format (srt or vtt)       | `srt`         |

</br></br>


### Model Size Guide

| Model      | Speed     | Accuracy | VRAM Usage | Best For                       |
|------------|-----------|----------|------------|--------------------------------|
| `tiny`     | Fastest   | Basic    | ~1GB       | Quick drafts, testing          |
| `base`     | Fast      | Good     | ~1GB       | General use, balanced          |
| `small`    | Moderate  | Better   | ~2GB       | Good quality, reasonable speed |
| `medium`   | Slow      | High     | ~5GB       | High quality needed            |
| `large-v2` | Very Slow | Highest  | ~10GB      | Professional work              |
| `large-v3` | Very Slow | Highest  | ~10GB      | Latest, best accuracy          |

</br></br>


## How It Works

1. **Video Validation**: Checks that the video file exists and is valid
2. **Audio Extraction**: Uses FFmpeg to extract audio in WAV format (16kHz, mono)
3. **Model Loading**: Loads the specified Whisper model with GPU acceleration
4. **Transcription**: Transcribes audio with timestamps using Voice Activity Detection
5. **Formatting**: Formats transcription into SRT or WebVTT subtitle format
6. **File Saving**: Saves subtitle file with the same name as the video

</br></br>


## Output

The application generates subtitle files with the same name as your video:
- Input: `my_video.mp4`
- Output: `my_video.srt` (or `my_video.vtt`)

</br></br>


## GPU Support

The application automatically detects CUDA-capable GPUs. If a GPU is detected:
- Uses `float16` precision for optimal performance
- Significantly faster processing times
- Lower latency for real-time applications

</br></br>


If no GPU is detected:
- Falls back to CPU processing
- Uses `int8` precision for efficiency
- Still produces accurate results (but slower)

</br></br>


## Performance Tips

1. **For fastest processing**: Use `tiny` or `base` model
2. **For best accuracy**: Use `large-v3` model (requires ~10GB VRAM)
3. **Balanced option**: Use `small` or `medium` model
4. **Multiple videos**: Process in batch by calling script multiple times
5. **Long videos**: Use smaller model sizes to avoid memory issues

</br></br>


## Project Structure

```
Voice to Text/
├── main.py                    # Main application entry point
├── subtitle_generator.py      # Subtitle generation with Whisper
├── video_processor.py         # Video processing and audio extraction
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

</br></br>


## License

This project uses the faster-whisper implementation of OpenAI's Whisper model.

</br></br>


## Acknowledgments

- OpenAI Whisper team for the amazing speech recognition model
- faster-whisper developers for the optimized implementation
- FFmpeg project for audio/video processing capabilities
