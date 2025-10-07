"""
Setup Validation Script

This script checks if all required dependencies are properly installed
    and configured for the Video to Subtitle application.

Classes:
    SetupValidator:
        Class to validate

Dependencies:
    - sys
        - For system-specific parameters and functions
    - subprocess
        - For running shell commands
"""

import sys
import subprocess
from typing import Tuple


class SetupValidator:
    """
    Validates the setup and configuration of the application.

    This class checks for required dependencies, GPU availability,
    and system configuration.

    Methods:
        __init__:
            Initialize the validator
        _check_python_version:
            Verify Python version is adequate
        _check_faster_whisper:
            Verify faster-whisper is installed
        _check_torch:
            Check if PyTorch is installed (optional)
        _check_ffmpeg:
            Verify FFmpeg is installed and accessible
        _check_gpu:
            Check GPU availability
        run_all_checks:
            Run all validation checks
    """

    def __init__(
        self
    ) -> None:
        """
        Initialize the SetupValidator.

        Args:
            None

        Returns:
            None
        """

        # Counters for summary
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0

    def _check_python_version(
        self
    ) -> bool:
        """
        Check if Python version meets requirements (3.8+).

        Args:
            None

        Returns:
            bool: True if version is adequate
        """

        print("Checking Python version...", end=" ")
        version_info = sys.version_info

        # Good Python
        if version_info >= (3, 8):
            print(f"✓ Python {version_info.major}.{version_info.minor}."
                  f"{version_info.micro}")
            self.checks_passed += 1
            return True

        # Naughty Python
        else:
            print(f"❌ Python {version_info.major}.{version_info.minor}."
                  f"{version_info.micro}")
            print("   ERROR: Python 3.8 or higher is required")
            self.checks_failed += 1
            return False

    def _check_faster_whisper(
        self
    ) -> bool:
        """
        Check if faster-whisper is installed.

        Args:
            None

        Returns:
            bool: True if installed
        """

        print("Checking faster-whisper...", end=" ")

        # Test importing faster-whisper
        try:
            import faster_whisper
            print(f"✓ Version {faster_whisper.__version__}")
            self.checks_passed += 1
            return True

        # Naughty faster-whisper
        except ImportError:
            print("❌ Not installed")
            print("   Install with: pip install faster-whisper")
            self.checks_failed += 1
            return False

    def _check_torch(
        self
    ) -> Tuple[bool, bool]:
        """
        Check if PyTorch is installed and CUDA availability.

        Returns:
            Tuple[bool, bool]: (torch_installed, cuda_available)
        """

        print("Checking PyTorch...", end=" ")

        # Try importing torch
        try:
            import torch
            print(f"✓ Version {torch.__version__}")

            # Check CUDA
            print("Checking CUDA availability...", end=" ")
            if torch.cuda.is_available():
                from torch import version as torch_version

                cuda_version = torch_version.cuda
                gpu_name = torch.cuda.get_device_name(0)

                print(f"✓ CUDA {cuda_version}")
                print(f"   GPU: {gpu_name}")

                self.checks_passed += 2
                return True, True

            # If no CUDA, fallback to CPU
            else:
                print("⚠ CUDA not available (CPU mode)")
                print("   The application will run on CPU (slower)")
                self.checks_passed += 1
                self.warnings += 1
                return True, False

        # Naughty torch! Couln't import it, so CPU will be used
        except ImportError:
            print("⚠ Not installed (optional)")
            print("   For better GPU detection, install PyTorch")
            self.warnings += 1
            return False, False

    def _check_ffmpeg(
        self
    ) -> bool:
        """
        Check if FFmpeg is installed and accessible.

        Args:
            None

        Returns:
            bool: True if FFmpeg is available
        """

        print("Checking FFmpeg...", end=" ")

        # Test run ffmpeg
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            # Extract version from output
            output = result.stdout.decode()
            version_line = output.split('\n')[0]
            print(f"✓ {version_line}")
            self.checks_passed += 1
            return True

        # If it fails, FFmpeg is not available
        except (
            subprocess.CalledProcessError,
            FileNotFoundError
        ):
            print("❌ ffmepg Not found")
            print("   Download from: https://ffmpeg.org/download.html")
            self.checks_failed += 1
            return False

    def _check_gpu(
        self
    ) -> bool:
        """
        Check GPU availability using faster-whisper's detection.

        Args:
            None

        Returns:
            bool: True if GPU is available
        """

        print("Checking GPU with faster-whisper...", end=" ")

        try:
            # Try importing faster-whisper
            from faster_whisper import WhisperModel

            # Try to initialize a model on GPU
            try:
                _ = WhisperModel(
                    "tiny",
                    device="cuda",
                    compute_type="float16"
                )
                print("✓ GPU initialization successful")
                self.checks_passed += 1
                return True

            # Couldn't initialize on GPU, fallback to CPU
            except Exception as e:
                print("⚠ GPU not available")
                print(f"   Reason: {str(e)}")
                print("   The application will run on CPU")
                self.warnings += 1
                return False

        # Naughty faster-whisper! Better install it first
        except ImportError:
            print("⚠ Cannot check (faster-whisper not installed)")
            self.warnings += 1
            return False

    def run_all_checks(
        self
    ) -> bool:
        """
        Run all validation checks and display summary.

        Returns:
            bool: True if all critical checks passed
        """

        print("=" * 60)
        print("SETUP VALIDATION")
        print("=" * 60)
        print()

        # Environment checks
        self._check_python_version()
        self._check_faster_whisper()
        self._check_ffmpeg()

        # GPU checks
        torch_installed, _ = self._check_torch()
        if torch_installed:
            self._check_gpu()

        # Display summary
        print()
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✓ Checks passed: {self.checks_passed}")
        print(f"❌ Checks failed: {self.checks_failed}")
        print(f"⚠ Warnings: {self.warnings}")
        print()

        if self.checks_failed == 0:
            print("✓ Setup is complete! You can run the application.")
            print()
            print("Quick start:")
            print("  python main.py your_video.mp4")
            return True

        else:
            print("❌ Setup is incomplete. Please install missing "
                  "dependencies.")
            print()
            return False


if __name__ == "__main__":
    validator = SetupValidator()
    success = validator.run_all_checks()

    sys.exit(0 if success else 1)
