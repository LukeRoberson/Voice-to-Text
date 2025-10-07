"""
Video to Subtitle Application

Main application for generating subtitles from MP4 video files using
the faster-whisper model with GPU acceleration.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

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
        process:
            Main processing method to generate subtitles
        _save_subtitle_file:
            Save subtitle content to file
    """

    def __init__(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        model_size: str = "base",
        subtitle_format: str = "srt"
    ) -> None:
        """
        Initialize the application with configuration parameters.

        Args:
            video_path (str): Path to the MP4 video file
            output_dir (str, optional): Directory for output files
            model_size (str): Whisper model size
                ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            subtitle_format (str): Subtitle format ('srt' or 'vtt')

        Returns:
            None
        """

        self.video_path = video_path
        self.output_dir = output_dir
        self.model_size = model_size
        self.subtitle_format = subtitle_format.lower()
        
        if self.subtitle_format not in ['srt', 'vtt']:
            raise ValueError(
                f"Invalid subtitle format: {subtitle_format}. "
                f"Must be 'srt' or 'vtt'"
            )

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

        Returns:
            None
        """

        print("=" * 60)
        print("VIDEO TO SUBTITLE GENERATOR")
        print("=" * 60)
        print()
        
        try:
            # Step 1: Initialize and validate video processor
            print("📋 Step 1: Validating video file...")
            video_processor = VideoProcessor(
                self.video_path,
                self.output_dir
            )
            
            if not video_processor.validate_video_file():
                print("❌ Video validation failed. Exiting.")
                return
            print()
            
            # Step 2: Extract audio from video
            print("📋 Step 2: Extracting audio from video...")
            audio_path = video_processor.extract_audio(
                output_format="wav",
                sample_rate=16000
            )
            print()
            
            # Step 3: Initialize subtitle generator with GPU support
            print("📋 Step 3: Initializing Whisper model...")
            subtitle_generator = SubtitleGenerator(
                model_size=self.model_size,
                device=None,  # Auto-detect GPU
                compute_type="float16"
            )
            print()
            
            # Step 4: Transcribe audio
            print("📋 Step 4: Transcribing audio...")
            segments = subtitle_generator.transcribe_audio(
                audio_path,
                language="en",
                beam_size=5,
                vad_filter=True
            )
            print()
            
            # Step 5: Format and save subtitles
            print("📋 Step 5: Generating subtitle file...")
            
            if self.subtitle_format == "srt":
                subtitle_content = subtitle_generator.format_srt(segments)
            else:  # vtt
                subtitle_content = subtitle_generator.format_vtt(segments)
            
            subtitle_path = self._save_subtitle_file(subtitle_content)
            print()
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print("🗑️  Removed temporary audio file")
            
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

        video_name = Path(self.video_path).stem
        subtitle_filename = f"{video_name}.{self.subtitle_format}"
        
        if self.output_dir:
            subtitle_path = os.path.join(
                self.output_dir,
                subtitle_filename
            )
        else:
            subtitle_path = os.path.join(
                Path(self.video_path).parent,
                subtitle_filename
            )
        
        try:
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ Subtitle file saved: {subtitle_path}")
            return subtitle_path
            
        except Exception as e:
            print(f"❌ Error saving subtitle file: {e}")
            raise


def main() -> None:
    """
    Main entry point for the application.

    Parses command-line arguments and initiates subtitle generation.

    Returns:
        None
    """

    parser = argparse.ArgumentParser(
        description="Generate subtitles from MP4 video files using "
                    "Whisper AI (GPU-accelerated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4
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
        help='Path to the input MP4 video file'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for subtitle file (default: same as video)'
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
        model_size=args.model_size,
        subtitle_format=args.format
    )
    
    app.process()


if __name__ == "__main__":
    main()
