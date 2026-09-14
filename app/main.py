"""
ReminisceCV — Application Entry Point
=======================================

Starts the ReminisceCV real-time OpenCV assistive memory system or performs
system diagnostic checks.

Usage
-----
Launch interactive OpenCV memory assistant:
    python main.py
    # or
    python -m app.main

Launch with mock feed (for environments without webcams):
    python main.py --mock

Run system diagnostic checks only:
    python main.py --check
"""

import argparse
import logging
import sys
from typing import Optional

from app import __project__, __version__
from app.config import config

logger = logging.getLogger("reminiscecv")


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


def run_checks() -> bool:
    """Perform system sanity checks and print diagnostics."""
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
        return False

    print()
    print("Configuration:")
    print(config.summary())

    print()
    print("All checks passed. ReminisceCV is ready.")
    print()
    return True


def main(argv: Optional[list[str]] = None) -> None:
    """Application entry point."""
    parser = argparse.ArgumentParser(
        description="ReminisceCV: Real-Time OpenCV Memory Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run system diagnostics and verify project imports without starting video feed",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic video capture and mock AI models (useful when no webcam is connected)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening an OpenCV GUI window",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Start with audio narration muted",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after processing N frames (0 = run continuously)",
    )

    args = parser.parse_args(argv)

    if args.check:
        ok = run_checks()
        if not ok:
            sys.exit(1)
        return

    _print_banner()
    print()
    print("Starting ReminisceCV Assistant...")
    print("Press 'Q' to quit, 'R' to reset recognition, 'M' to mute/unmute.")
    print()

    from scripts.run_assistant import build_app

    app = build_app(
        camera_index=args.camera,
        use_mock=args.mock,
        start_muted=args.mute,
    )

    try:
        app.run(
            max_frames=args.max_frames if args.max_frames > 0 else None,
            headless=args.headless,
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error("Error during execution: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
