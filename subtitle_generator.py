"""
Subtitle Generator Module

This module provides functionality to generate subtitles from audio using
    the faster-whisper model optimized for GPU acceleration.

Uses float16 precision (by default) for faster inference on compatible GPUs.
    Uses int8 precision when running on CPU (slower).

Classes:
    SubtitleGenerator:
        Class for loading the Whisper model and generating subtitles
"""

from typing import List, Dict, Optional
from faster_whisper import WhisperModel
import torch


class SubtitleGenerator:
    """
    A class for generating subtitles using the faster-whisper model.

    This class leverages GPU acceleration when available and provides
    methods to transcribe audio and format the output as subtitle files.

    Attributes:
        model_size (str): Size of the Whisper model to use
        device (str): Device to run the model on ('cuda' or 'cpu')
        compute_type (str): Computation precision type

    Methods:
        __init__:
            Initialize the SubtitleGenerator
        _check_gpu_availability:
            Check if CUDA GPU is available
        _format_timestamp_srt:
            Format timestamp for SRT format
        _format_timestamp_vtt:
            Format timestamp for WebVTT format
        transcribe_audio:
            Transcribe audio file to text with timestamps
        format_srt:
            Format transcription segments into SRT subtitle format
        format_vtt:
            Format transcription segments into WebVTT subtitle format
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float16"
    ) -> None:
        """
        Initialize the SubtitleGenerator with specified model parameters.

        Args:
            model_size (str): Whisper model size
                ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            device (str, optional): Device to use ('cuda' or 'cpu').
                If None, automatically detects GPU availability
            compute_type (str): Precision type for computation
                ('float16', 'int8_float16', 'int8')

        Returns:
            None
        """

        # Get and check the model size
        self.model_size = model_size
        if model_size not in [
            "tiny", "base", "small", "medium", "large-v2", "large-v3"
        ]:
            raise ValueError(
                f"Invalid model size: {model_size}. "
                "Choose from 'tiny', 'base', 'small', "
                "'medium', 'large-v2', 'large-v3'."
            )

        # Automatically detect device if not specified
        if device is None:
            self.device = "cuda" if self._check_gpu_availability() else "cpu"
            print(f"✓ Auto-detected device: {self.device}")
        else:
            self.device = device

        # Set precision for CPU
        if self.device == "cpu":
            self.compute_type = "int8"
            print("ℹ Using int8 compute type for CPU")

        # Set precision for GPU
        else:
            self.compute_type = compute_type

            # Confirm valid compute type (precision)
            if compute_type not in ["float16", "int8_float16", "int8"]:
                raise ValueError(
                    f"Invalid compute type: {compute_type}. "
                    "Choose from 'float16', 'int8_float16', 'int8'."
                )
            else:
                print(f"✓ Using {compute_type} compute type for GPU")

        # Load the Whisper model
        try:
            print(f"📥 Loading Whisper model: {model_size}...")
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            print(f"✓ Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _check_gpu_availability(
        self
    ) -> bool:
        """
        Check if CUDA-capable GPU is available.

        Args:
            None

        Returns:
            bool: True if CUDA GPU is available, False otherwise
        """

        # Use pytorch to check for CUDA availability
        try:
            return torch.cuda.is_available()

        # CUDA not available or torch not installed
        except Exception:
            return False

    def _format_timestamp_srt(
        self,
        seconds: float
    ) -> str:
        """
        Format timestamp for SRT format (HH:MM:SS,mmm).

        Args:
            seconds (float): Time in seconds

        Returns:
            str: Formatted timestamp string
        """

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _format_timestamp_vtt(
        self,
        seconds: float
    ) -> str:
        """
        Format timestamp for WebVTT format (HH:MM:SS.mmm).

        Args:
            seconds (float): Time in seconds

        Returns:
            str: Formatted timestamp string
        """

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    def transcribe_audio(
        self,
        audio_path: str,
        language: str = "en",
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> List[Dict]:
        """
        Transcribe audio file to text with timestamps.

        The 'beam size', aka 'beam width', controls the number of 'paths' that
            are explored during decoding.
        A larger beam size generally improves accuracy but increases the time
            taken to transcribe.
        Values are from 1 to 10 (default is 5).

        Args:
            audio_path (str): Path to the audio file
            language (str): Language code (default: 'en' for English)
            beam_size (int): Beam size for decoding (higher = more accurate
                but slower)
            vad_filter (bool): Use Voice Activity Detection to filter
                non-speech segments

        Returns:
            List[Dict]: List of transcription segments with timestamps
                Each segment contains: start, end, text
        """

        # Validate beam size
        if not (1 <= beam_size <= 10):
            raise ValueError("Beam size must be between 1 and 10")

        print(f"🎤 Transcribing audio: {audio_path}")
        print(f"   Language: {language}, Beam size: {beam_size}, "
              f"VAD filter: {vad_filter}")

        # Get the model to begin transcription
        try:
            # Segments are a portion of the audio with start/end times
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=False
            )

            print(f"✓ Detected language: {info.language} "
                  f"(probability: {info.language_probability:.2f})")

            # Convert segments generator to list of dictionaries
            transcription_segments = []
            for segment in segments:
                transcription_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                })

            print(
                f"✓ Transcription complete: "
                f"{len(transcription_segments)} segments"
            )

            return transcription_segments

        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            raise

    def format_srt(
        self,
        segments: List[Dict]
    ) -> str:
        """
        Format transcription segments into SRT subtitle format.

        SRT format structure:
        1
        00:00:00,000 --> 00:00:02,000
        Subtitle text here

        Args:
            segments (List[Dict]): List of transcription segments
                with 'start', 'end', and 'text' keys

        Returns:
            str: Formatted SRT subtitle content
        """

        srt_content = []

        # Loop over each segment
        for i, segment in enumerate(segments, 1):
            # Time at top, and text below
            start_time = self._format_timestamp_srt(segment['start'])
            end_time = self._format_timestamp_srt(segment['end'])
            text = segment['text']

            # Append the formatted segment to the list
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(f"{text}")
            srt_content.append("")  # Empty line between segments

        # Combine all segments into a single string
        return "\n".join(srt_content)

    def format_vtt(
        self,
        segments: List[Dict]
    ) -> str:
        """
        Format transcription segments into WebVTT subtitle format.

        WebVTT format structure:
        WEBVTT

        00:00:00.000 --> 00:00:02.000
        Subtitle text here

        Args:
            segments (List[Dict]): List of transcription segments
                with 'start', 'end', and 'text' keys

        Returns:
            str: Formatted WebVTT subtitle content
        """

        # Initialize VTT content with header
        vtt_content = ["WEBVTT", ""]

        # Loop over each segment
        for segment in segments:
            # Time at top, and text below
            start_time = self._format_timestamp_vtt(segment['start'])
            end_time = self._format_timestamp_vtt(segment['end'])
            text = segment['text']

            # Append the formatted segment to the list
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(f"{text}")
            vtt_content.append("")  # Empty line between segments

        # Combine all segments into a single string
        return "\n".join(vtt_content)
