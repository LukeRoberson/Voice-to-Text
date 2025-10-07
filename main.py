"""
Video to Subtitle Application

Main application for generating subtitles from MP4 video files using
the faster-whisper model with GPU acceleration.

Classes:
    VideoSubtitleApp:
        Main application class to orchestrate video processing and subtitle
        generation.

Functions:
    main:
        Entry point for the application.

Dependencies:
    - argparse
        - For command-line argument parsing
    - os, pathlib
        - For file and path operations
    - sys
        - For system-specific parameters and functions
    - tempfile
        - For creating temporary files
    - requests
        - For downloading video files from URLs
    - urllib.parse
        - For URL parsing

Custom Modules:
    - video_processor
        - For video validation and audio extraction
    - subtitle_generator
        - For transcription and subtitle formatting
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import tempfile
import requests
from urllib.parse import urlparse

from video_processor import VideoProcessor
from subtitle_generator import SubtitleGenerator


class VideoSubtitleApp:
    """
    Main application class for video to subtitle conversion.

    This class orchestrates the video processing and subtitle generation
    workflow, providing a simple interface for the entire process.

    Attributes:
        video_path (str): Path to the input video file
        output_dir (str): Directory for output files
        model_size (str): Whisper model size to use
        subtitle_format (str): Output subtitle format

    Methods:
        __init__:
            Initialize the VideoSubtitleApp
        _is_url:
            Check if the given path is a URL
        _download_video:
            Download video from URL to a temporary file
        _save_subtitle_file:
            Save subtitle content to file
        process:
            Main processing method to generate subtitles
    """

    def __init__(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        model_size: str = "base",
        subtitle_format: str = "srt"
    ) -> None:
        """
        Initialize the application with configuration parameters.

        Args:
            video_path (str):
                Path to the MP4 video file
            output_dir (str, optional):
                Directory for output files
            output_filename (str, optional):
                Output filename stem for subtitle file
            model_size (str): Whisper model size
                ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            subtitle_format (str):
                Subtitle format ('srt' or 'vtt')

        Returns:
            None
        """

        # Config
        self.video_path = video_path
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.model_size = model_size
        self.subtitle_format = subtitle_format.lower()

        # Validate subtitle format
        if self.subtitle_format not in ['srt', 'vtt']:
            raise ValueError(
                f"Invalid subtitle format: {subtitle_format}. "
                f"Must be 'srt' or 'vtt'"
            )

    def _is_url(
        self,
        path: str
    ) -> bool:
        """
        Check if the given path is a URL.

        Args:
            path (str): Path or URL

        Returns:
            bool: True if path is a URL, False otherwise
        """

        parsed = urlparse(path)
        return parsed.scheme in ('http', 'https')

    def _download_video(
        self,
        url: str
    ) -> str:
        """
        Download video from URL to a temporary file.

        Args:
            url (str): Video URL

        Returns:
            str: Path to the downloaded temporary file
        """

        print(f"⬇️  Downloading video from URL: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        suffix = os.path.splitext(urlparse(url).path)[-1] or ".mp4"
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()

        print(f"✓ Downloaded to temporary file: {temp_file.name}")
        return temp_file.name

    def _save_subtitle_file(
        self,
        content: str
    ) -> str:
        """
        Save subtitle content to file.

        Args:
            content (str): Formatted subtitle content

        Returns:
            str: Path to the saved subtitle file
        """

        # Use custom output filename if provided
        if self.output_filename:
            subtitle_filename = (
                self.output_filename + f".{self.subtitle_format}"
            )

        # Use video name as default
        else:
            video_name = Path(self.video_path).stem
            subtitle_filename = f"{video_name}.{self.subtitle_format}"

        # Write to the given output directory
        if self.output_dir:
            subtitle_path = os.path.join(
                self.output_dir,
                subtitle_filename
            )

        # Or, write to the same directory as the video file
        else:
            subtitle_path = os.path.join(
                Path(self.video_path).parent,
                subtitle_filename
            )

        # Write the content to a file
        try:
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✓ Subtitle file saved: {subtitle_path}")
            return subtitle_path

        except Exception as e:
            print(f"❌ Error saving subtitle file: {e}")
            raise

    def process(
        self
    ) -> None:
        """
        Process the video and generate subtitles.

        This method coordinates the entire workflow:
            1. Validate video file
            2. Extract audio from video
            3. Load Whisper model
            4. Transcribe audio
            5. Generate and save subtitle file

        Args:
            None

        Returns:
            None
        """

        print("=" * 60)
        print("VIDEO TO SUBTITLE GENERATOR")
        print("=" * 60)
        print()

        temp_video_path = None
        try:
            # Step 1: Handle URL input
            if self._is_url(self.video_path):
                temp_video_path = self._download_video(self.video_path)
                video_path = temp_video_path

                # Set output dir to local dir if not set
                if not self.output_dir:
                    self.output_dir = str(Path("."))

                # Set filename if not set
                if not self.output_filename:
                    self.output_filename = (
                        self.video_path.split("/")[-1].rsplit(".", 1)[0]
                    )
                    print(f"Using output filename: {self.output_filename}")

            else:
                video_path = self.video_path

            # Step 2: Initialize and validate video processor
            print("📋 Step 1: Validating video file...")
            video_processor = VideoProcessor(
                video_path,
                self.output_dir
            )

            if not video_processor.validate_video_file():
                print("❌ Video validation failed. Exiting.")
                return
            print()

            # Step 3: Extract audio from video
            print("📋 Step 2: Extracting audio from video...")
            audio_path = video_processor.extract_audio(
                output_format="wav",
                sample_rate=16000
            )
            print()

            # Step 4: Initialize subtitle generator with GPU support
            print("📋 Step 3: Initializing Whisper model...")
            subtitle_generator = SubtitleGenerator(
                model_size=self.model_size,
                device=None,  # Auto-detect GPU
                compute_type="float16"
            )
            print()

            # Step 45: Transcribe audio
            print("📋 Step 4: Transcribing audio...")
            segments = subtitle_generator.transcribe_audio(
                audio_path,
                language="en",
                beam_size=5,
                vad_filter=True
            )
            print()

            # Step 6: Format and save subtitles
            print("📋 Step 5: Generating subtitle file...")

            # Generate srt file
            if self.subtitle_format == "srt":
                subtitle_content = subtitle_generator.format_srt(segments)

            # Or, generate vtt file
            else:
                subtitle_content = subtitle_generator.format_vtt(segments)

            # Save subtitle file
            subtitle_path = self._save_subtitle_file(subtitle_content)
            print()

            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print("🗑️  Removed temporary audio file")

            # Clean up temporary video file if downloaded
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print("🗑️  Removed temporary downloaded video file")

            print()
            print("=" * 60)
            print("✓ SUBTITLE GENERATION COMPLETE!")
            print(f"📄 Subtitle file: {subtitle_path}")
            print("=" * 60)

        except Exception as e:
            print()
            print("=" * 60)
            print(f"❌ ERROR: {e}")
            print("=" * 60)
            sys.exit(1)


def main() -> None:
    """
    Main entry point for the application.

    1. Creates argument parser
    2. Creates an instance of VideoSubtitleApp
    3. Runs the app

    Returns:
        None
    """

    # CLI arg parser
    parser = argparse.ArgumentParser(
        description="Generate subtitles from MP4 video files using "
                    "Whisper AI (GPU-accelerated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python main.py video.mp4
            python main.py video.mp4 -o ./subtitles -n my_video
            python main.py video.mp4 -m medium -f vtt
            python main.py video.mp4 -o ./subtitles -m large-v3

            Model sizes:
            tiny    - Fastest, least accurate (~1GB VRAM)
            base    - Fast, good accuracy (~1GB VRAM)
            small   - Balanced (~2GB VRAM)
            medium  - High accuracy (~5GB VRAM)
            large-v2/large-v3 - Best accuracy (~10GB VRAM)
        """
    )

    parser.add_argument(
        'video_path',
        type=str,
        help=(
            'Path to the input MP4 video file. '
            'A local mp4 file path or a URL to an mp4 file is accepted.'
        )
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for subtitle file (default: same as video)'
    )

    parser.add_argument(
        '-n', '--output-name',
        type=str,
        default=None,
        help=(
            'Output filename stem for subtitle file, without extension '
            '(default: same as video with srt or vtt extension)'
        )
    )

    parser.add_argument(
        '-m', '--model-size',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
        help='Whisper model size (default: base)'
    )

    parser.add_argument(
        '-f', '--format',
        type=str,
        default='srt',
        choices=['srt', 'vtt'],
        help='Subtitle format (default: srt)'
    )

    args = parser.parse_args()

    # Create and run the application
    app = VideoSubtitleApp(
        video_path=args.video_path,
        output_dir=args.output_dir,
        output_filename=args.output_name,
        model_size=args.model_size,
        subtitle_format=args.format
    )

    # Run the app
    app.process()


if __name__ == "__main__":
    main()
