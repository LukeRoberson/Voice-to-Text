"""
Video Processor Module

This module provides functionality to extract audio from video files
for subtitle generation.
"""

from typing import Optional
import os
import subprocess
from pathlib import Path


class VideoProcessor:
    """
    A class for processing video files and extracting audio.

    This class handles video file validation and audio extraction
    using FFmpeg for compatibility with the subtitle generator.

    Attributes:
        video_path (str): Path to the video file
        output_dir (str): Directory for output files

    Methods:
        __init__:
            Initialize the VideoProcessor
        validate_video_file:
            Validate that the video file exists and is accessible
        extract_audio:
            Extract audio from video file
        _check_ffmpeg:
            Check if FFmpeg is available in the system
    """

    def __init__(
        self,
        video_path: str,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the VideoProcessor with a video file.

        Args:
            video_path (str): Path to the video file (MP4 format)
            output_dir (str, optional): Directory for output files.
                If None, uses the same directory as the video file

        Returns:
            None
        """

        self.video_path = video_path
        
        if output_dir is None:
            self.output_dir = str(Path(video_path).parent)
        else:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"📹 Video file: {video_path}")
        print(f"📁 Output directory: {self.output_dir}")

    def validate_video_file(
        self
    ) -> bool:
        """
        Validate that the video file exists and is accessible.

        Returns:
            bool: True if valid, False otherwise
        """

        if not os.path.exists(self.video_path):
            print(f"❌ Video file not found: {self.video_path}")
            return False
        
        if not os.path.isfile(self.video_path):
            print(f"❌ Path is not a file: {self.video_path}")
            return False
        
        # Check file extension
        valid_extensions = ['.mp4', '.MP4']
        file_ext = Path(self.video_path).suffix
        if file_ext not in valid_extensions:
            print(f"❌ Invalid file extension: {file_ext}. "
                  f"Expected: {', '.join(valid_extensions)}")
            return False
        
        print("✓ Video file validated successfully")
        return True

    def extract_audio(
        self,
        output_format: str = "wav",
        sample_rate: int = 16000
    ) -> str:
        """
        Extract audio from video file using FFmpeg.

        Args:
            output_format (str): Audio format ('wav', 'mp3', etc.)
            sample_rate (int): Audio sample rate in Hz
                (16000 is optimal for Whisper)

        Returns:
            str: Path to the extracted audio file
        """

        if not self._check_ffmpeg():
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and "
                "add it to your system PATH."
            )
        
        # Generate output audio filename
        video_name = Path(self.video_path).stem
        audio_filename = f"{video_name}_audio.{output_format}"
        audio_path = os.path.join(self.output_dir, audio_filename)
        
        print(f"🎵 Extracting audio to: {audio_path}")
        
        try:
            # FFmpeg command to extract audio
            codec = 'pcm_s16le' if output_format == 'wav' else 'libmp3lame'
            command = [
                'ffmpeg',
                '-i', self.video_path,
                '-vn',  # No video
                '-acodec', codec,
                '-ar', str(sample_rate),  # Sample rate
                '-ac', '1',  # Mono channel
                '-y',  # Overwrite output file
                audio_path
            ]
            
            # Run FFmpeg with suppressed output
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            print("✓ Audio extracted successfully")
            return audio_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e.stderr.decode()}")
            raise
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            raise

    def _check_ffmpeg(
        self
    ) -> bool:
        """
        Check if FFmpeg is available in the system PATH.

        Returns:
            bool: True if FFmpeg is available, False otherwise
        """

        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
