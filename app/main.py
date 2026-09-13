"""
ReminisceCV — Application Entry Point
=======================================

Minimal startup that verifies the project structure is intact and
all packages are importable.  Feature logic is added in later stages.

Usage
-----
    python -m app.main
    # or
    python main.py          (via the wrapper at project root)
"""

import sys

from app import __project__, __version__
from app.config import config


def _print_banner() -> None:
    """Print a startup banner with project information."""
    width = 58
    print("=" * width)
    print(f"  {config.app_name} v{config.app_version}")
    print(f"  A Real-Time Vision-Based Personal Memory Retrieval")
    print(f"  and Assistive System")
    print("=" * width)


def _check_directories() -> None:
    """Ensure required directories exist, create them if missing."""
    config.ensure_directories()
    for dir_path in (config.data_dir, config.models_dir):
        print(f"  [OK] {dir_path.relative_to(config.project_root)}/")


def _verify_imports() -> bool:
    """Import each sub-package to confirm the structure is valid."""
    sub_packages = [
        "app.database",
        "app.vision",
        "app.recognition",
        "app.memory",
        "app.speech",
        "app.ui",
    ]
    all_ok = True
    for pkg in sub_packages:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError as e:
            print(f"  [FAIL] {pkg}: {e}")
            all_ok = False
    return all_ok


def main() -> None:
    """Application entry point."""
    _print_banner()

    print()
    print("Checking project structure...")
    _check_directories()

    print()
    print("Verifying package imports...")
    all_ok = _verify_imports()

    if not all_ok:
        print()
        print("Some checks failed. Fix import errors before continuing.")
        sys.exit(1)

    print()
    print("Configuration:")
    print(config.summary())

    print()
    print("All checks passed. ReminisceCV is ready for development.")
    print()


if __name__ == "__main__":
    main()
