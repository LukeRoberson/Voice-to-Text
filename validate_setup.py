"""
Setup Validation Script

This script checks if all required dependencies are properly installed
and configured for the Video to Subtitle application.
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
        check_python_version:
            Verify Python version is adequate
        check_faster_whisper:
            Verify faster-whisper is installed
        check_torch:
            Check if PyTorch is installed (optional)
        check_ffmpeg:
            Verify FFmpeg is installed and accessible
        check_gpu:
            Check GPU availability
        run_all_checks:
            Run all validation checks
    """

    def __init__(
        self
    ) -> None:
        """
        Initialize the SetupValidator.

        Returns:
            None
        """

        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0

    def check_python_version(
        self
    ) -> bool:
        """
        Check if Python version meets requirements (3.8+).

        Returns:
            bool: True if version is adequate
        """

        print("Checking Python version...", end=" ")
        version_info = sys.version_info
        
        if version_info >= (3, 8):
            print(f"✓ Python {version_info.major}.{version_info.minor}."
                  f"{version_info.micro}")
            self.checks_passed += 1
            return True
        else:
            print(f"❌ Python {version_info.major}.{version_info.minor}."
                  f"{version_info.micro}")
            print("   ERROR: Python 3.8 or higher is required")
            self.checks_failed += 1
            return False

    def check_faster_whisper(
        self
    ) -> bool:
        """
        Check if faster-whisper is installed.

        Returns:
            bool: True if installed
        """

        print("Checking faster-whisper...", end=" ")
        
        try:
            import faster_whisper
            print(f"✓ Version {faster_whisper.__version__}")
            self.checks_passed += 1
            return True
        except ImportError:
            print("❌ Not installed")
            print("   Install with: pip install faster-whisper")
            self.checks_failed += 1
            return False

    def check_torch(
        self
    ) -> Tuple[bool, bool]:
        """
        Check if PyTorch is installed and CUDA availability.

        Returns:
            Tuple[bool, bool]: (torch_installed, cuda_available)
        """

        print("Checking PyTorch...", end=" ")
        
        try:
            import torch
            print(f"✓ Version {torch.__version__}")
            
            # Check CUDA
            print("Checking CUDA availability...", end=" ")
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✓ CUDA {cuda_version}")
                print(f"   GPU: {gpu_name}")
                self.checks_passed += 2
                return True, True
            else:
                print("⚠ CUDA not available (CPU mode)")
                print("   The application will run on CPU (slower)")
                self.checks_passed += 1
                self.warnings += 1
                return True, False
                
        except ImportError:
            print("⚠ Not installed (optional)")
            print("   For better GPU detection, install PyTorch:")
            print("   pip install torch torchvision torchaudio "
                  "--index-url https://download.pytorch.org/whl/cu118")
            self.warnings += 1
            return False, False

    def check_ffmpeg(
        self
    ) -> bool:
        """
        Check if FFmpeg is installed and accessible.

        Returns:
            bool: True if FFmpeg is available
        """

        print("Checking FFmpeg...", end=" ")
        
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
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Not found")
            print("   Install FFmpeg:")
            print("   - Windows: choco install ffmpeg")
            print("   - Or download from: https://ffmpeg.org/download.html")
            self.checks_failed += 1
            return False

    def check_gpu(
        self
    ) -> bool:
        """
        Check GPU availability using faster-whisper's detection.

        Returns:
            bool: True if GPU is available
        """

        print("Checking GPU with faster-whisper...", end=" ")
        
        try:
            from faster_whisper import WhisperModel
            
            # Try to initialize a model on GPU
            try:
                model = WhisperModel(
                    "tiny",
                    device="cuda",
                    compute_type="float16"
                )
                print("✓ GPU initialization successful")
                self.checks_passed += 1
                return True
            except Exception as e:
                print("⚠ GPU not available")
                print(f"   Reason: {str(e)}")
                print("   The application will run on CPU")
                self.warnings += 1
                return False
                
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
        
        # Run all checks
        self.check_python_version()
        self.check_faster_whisper()
        torch_installed, cuda_available = self.check_torch()
        self.check_ffmpeg()
        
        if torch_installed:
            self.check_gpu()
        
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
            print("See QUICKSTART.md for installation instructions.")
            return False


def main() -> None:
    """
    Main entry point for the validation script.

    Returns:
        None
    """

    validator = SetupValidator()
    success = validator.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
