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
from app.config import APP_NAME, DATA_DIR, MODELS_DIR, DB_PATH


def _print_banner() -> None:
    """Print a startup banner with project information."""
    width = 58
    print("=" * width)
    print(f"  {APP_NAME} v{__version__}")
    print(f"  A Real-Time Vision-Based Personal Memory Retrieval")
    print(f"  and Assistive System")
    print("=" * width)


def _check_directories() -> None:
    """Ensure required directories exist, create them if missing."""
    for dir_path in (DATA_DIR, MODELS_DIR):
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {dir_path.relative_to(DATA_DIR.parent)}/")


def main() -> None:
    """Application entry point."""
    _print_banner()

    print()
    print("Checking project structure...")
    _check_directories()

    print()
    print("Verifying package imports...")
    # Import each sub-package to confirm the structure is valid
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

    print()
    if all_ok:
        print("All checks passed. ReminisceCV is ready for development.")
        print()
        print("Next steps:")
        print("  - Stage 2: SQLite memory database")
        print("  - Stage 3: CLIP object embedding pipeline")
        print("  - Stage 4: Face detection & embedding")
    else:
        print("Some checks failed. Fix import errors before continuing.")
        sys.exit(1)

    print()
    print(f"Database path: {DB_PATH}")
    print(f"Models path:   {MODELS_DIR}")
    print()


if __name__ == "__main__":
    main()
