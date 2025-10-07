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
from tqdm import tqdm

from video_processor import VideoProcessor
from subtitle_generator import SubtitleGenerator
import csv


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
        subtitle_format: str = "srt",
        csv_file: Optional[str] = None
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
            csv_file (str, optional):
                CSV file with list of videos to process in a batch

        Returns:
            None
        """

        # Config
        self.video_path = video_path
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.model_size = model_size
        self.subtitle_format = subtitle_format.lower()
        self.csv_file = csv_file

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

        response = requests.get(
            url,
            stream=True
        )
        response.raise_for_status()

        # Get total file size from headers
        total_size = int(response.headers.get('content-length', 0))

        # Determine file extension
        suffix = os.path.splitext(urlparse(url).path)[-1] or ".mp4"

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        # Download with progress bar
        with tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc='Downloading'
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
                pbar.update(len(chunk))

        temp_file.close()

        print(f"✓ Downloaded to temporary file: {temp_file.name}")
        return temp_file.name

    def _save_subtitle_file(
        self,
        content: str,
        video_name: str,
        video_path: str
    ) -> str:
        """
        Save subtitle content to file.

        Handles three scenarios:
        1. Single file with custom output filename
        2. Single file using video filename
        3. Batch processing from CSV with video names

        Args:
            content (str): Formatted subtitle content
            video_name (str): Name to use for the subtitle file
            video_path (str): Path or URL of the video file

        Returns:
            str: Path to the saved subtitle file
        """

        # Create subtitle filename
        subtitle_filename = f"{video_name}.{self.subtitle_format}"

        # Determine output directory
        if self.output_dir:
            # Use specified output directory
            output_dir = self.output_dir
        elif self._is_url(video_path):
            # For URLs, use current working directory
            output_dir = os.getcwd()
        else:
            # For local files, use same directory as video
            output_dir = str(Path(video_path).parent)

        # Construct full path
        subtitle_path = os.path.join(
            output_dir,
            subtitle_filename
        )

        # Write the content to a file
        try:
            with open(
                subtitle_path,
                'w',
                encoding='utf-8'
            ) as f:
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
            1. Load video list from CSV or create single-item list
            2. For each video:
                a. Validate video file or download from URL
                b. Extract audio from video
                c. Load Whisper model (once, reused for all videos)
                d. Transcribe audio
                e. Generate and save subtitle file
                f. Clean up temporary files

        Args:
            None

        Returns:
            None
        """

        print("=" * 60)
        print("VIDEO TO SUBTITLE GENERATOR")
        print("=" * 60)
        print()

        # Handle a list of videos from CSV file
        if self.csv_file:
            print(f"📋 Processing batch from CSV file: {self.csv_file}")

            try:
                with open(
                    self.csv_file,
                    mode='r',
                    encoding='utf-8'
                ) as csvfile:
                    reader = csv.DictReader(csvfile)
                    videos = [row for row in reader]
                print(f"Found {len(videos)} videos in CSV.")

            except Exception as e:
                print(f"❌ Failed to read CSV file: {e}")
                sys.exit(1)

        # Or, create a list of the one video to process
        else:
            videos = [
                {
                    'name': self.output_filename or Path(self.video_path).stem,
                    'file': self.video_path
                }
            ]

        # Initialize subtitle generator once (reuse for all videos)
        print("📋 Initializing Whisper model...")
        subtitle_generator = SubtitleGenerator(
            model_size=self.model_size,
            device=None,  # Auto-detect GPU
            compute_type="float16"
        )
        print()

        # Process each video in the list
        for idx, video in enumerate(videos, start=1):
            temp_video_path = None
            audio_path = None
            temp_video_path = None

            try:
                print("=" * 60)
                print(f"Processing video {idx}/{len(videos)}: {video['name']}")
                print("=" * 60)
                print()

                video_file = video['file']
                video_name = video['name']

                # Handle URL input or local file
                if self._is_url(video_file):
                    temp_video_path = self._download_video(video_file)
                    video_path = temp_video_path
                else:
                    video_path = video_file

                # Step 1: Initialize and validate video processor
                print("📋 Step 1: Validating video file...")
                video_processor = VideoProcessor(
                    video_path,
                    self.output_dir
                )

                if not video_processor.validate_video_file():
                    print(
                        f"❌ Video validation failed for {video_name}. "
                        f"Skipping."
                    )
                    continue
                print()

                # Step 2: Extract audio from video
                print("📋 Step 2: Extracting audio from video...")
                audio_path = video_processor.extract_audio(
                    output_format="wav",
                    sample_rate=16000
                )
                print()

                # Step 3: Transcribe audio
                print("📋 Step 3: Transcribing audio...")
                segments = subtitle_generator.transcribe_audio(
                    audio_path,
                    language="en",
                    beam_size=5,
                    vad_filter=True
                )
                print()

                # Step 4: Format and save subtitles
                print("📋 Step 4: Generating subtitle file...")

                # Store original output filename
                original_output_filename = self.output_filename

                # Use video name from CSV or original filename
                self.output_filename = video_name

                # Generate srt file
                if self.subtitle_format == "srt":
                    subtitle_content = subtitle_generator.format_srt(segments)

                # Or, generate vtt file
                else:
                    subtitle_content = subtitle_generator.format_vtt(segments)

                # Save subtitle file
                subtitle_path = self._save_subtitle_file(
                    subtitle_content,
                    video_name,
                    video_file
                )

                # Restore original output filename
                self.output_filename = original_output_filename
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
                print(f"❌ ERROR processing {video['name']}: {e}")
                print("=" * 60)
                print()

                # Clean up on error
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass

                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except Exception:
                        pass

                # Continue to next video instead of exiting
                continue


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
        description=(
            "Generate subtitles from MP4 video files using "
            "Whisper AI (GPU-accelerated)"
        ),
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
        nargs='?',
        default=None,
        help=(
            'Path to the input MP4 video file. '
            'A local mp4 file path or a URL to an mp4 file is accepted. '
            'Not required if --csv-file is provided.'
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

    parser.add_argument(
        '-c', '--csv-file',
        type=str,
        default=None,
        help=(
            'CSV file with list of videos to process in a batch. '
            'Overrides the video_path and output file arguments. '
            'Expects a name and file column. Other columns are ignored.'
        )
    )

    args = parser.parse_args()

    # Validate that either video_path or csv_file is provided
    if not args.video_path and not args.csv_file:
        parser.error(
            "Either video_path or --csv-file must be provided"
        )

    # Create and run the application
    app = VideoSubtitleApp(
        video_path=args.video_path,
        output_dir=args.output_dir,
        output_filename=args.output_name,
        model_size=args.model_size,
        subtitle_format=args.format,
        csv_file=args.csv_file
    )

    # Run the app
    app.process()


if __name__ == "__main__":
    main()
