"""
scripts/demo_webcam_feed.py — Demo: Real-Time Webcam Feed Capture
================================================================

Displays live video from the webcam using ReminisceCV's CameraService:
1. Validates webcam hardware availability.
2. Configures resolution (using config or CLI overrides).
3. Reads frames in real time with FPS estimation.
4. Overlays capture statistics (resolution, FPS, frame count).
5. Cleanly shuts down on pressing 'q' or 'ESC'.

Usage
-----
Standard real-time webcam feed:
    python scripts/demo_webcam_feed.py

Custom camera index and resolution:
    python scripts/demo_webcam_feed.py --camera-index 0 --width 640 --height 480

Headless verification (e.g. CI / testing capture logic without a GUI):
    python scripts/demo_webcam_feed.py --headless --max-frames 30
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.vision import (
    CameraError,
    CameraNotFoundError,
    CameraService,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo_webcam_feed")


class SyntheticVideoCapture:
    """Mock capture device generating animated test frames for headless/demo environments."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._opened = True
        self._frame_idx = 0

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        if not self._opened:
            return False, np.array([])
        self._frame_idx += 1
        # Create a synthetic gradient canvas with an animated moving circle
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :, 0] = np.linspace(30, 80, self.height, dtype=np.uint8)[:, None]
        frame[:, :, 1] = np.linspace(40, 100, self.height, dtype=np.uint8)[:, None]
        frame[:, :, 2] = np.linspace(50, 120, self.height, dtype=np.uint8)[:, None]

        # Moving circle
        cx = int((np.sin(self._frame_idx * 0.08) * 0.35 + 0.5) * self.width)
        cy = int((np.cos(self._frame_idx * 0.08) * 0.3 + 0.5) * self.height)
        cv2.circle(frame, (cx, cy), 45, (60, 220, 100), -1)
        cv2.putText(
            frame,
            "SYNTHETIC CAMERA FEED (MOCK)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        time.sleep(0.03)  # Emulate ~30 FPS
        return True, frame

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return True

    def release(self) -> None:
        self._opened = False


def run_camera_feed(
    camera_index: int,
    width: int,
    height: int,
    auto_resize: bool = True,
    headless: bool = False,
    max_frames: int = 0,
    use_mock: bool = False,
) -> int:
    """Run the camera capture loop.

    Parameters
    ----------
    camera_index : int
        Webcam device index.
    width : int
        Target frame width in pixels.
    height : int
        Target frame height in pixels.
    auto_resize : bool
        Whether to enforce exact target size via resizing.
    headless : bool
        If True, runs without cv2.imshow GUI display.
    max_frames : int
        Exit after this many frames (0 = run indefinitely until quit).
    use_mock : bool
        Force use of synthetic capture device.

    Returns
    -------
    int
        Exit code (0 on success, non-zero on error).
    """
    logger.info("Initializing ReminisceCV CameraService...")
    logger.info(
        "Configuration: index=%d, resolution=%dx%d, auto_resize=%s, headless=%s",
        camera_index,
        width,
        height,
        auto_resize,
        headless,
    )

    capture_device = None
    if use_mock:
        logger.info("Using SyntheticVideoCapture device.")
        capture_device = SyntheticVideoCapture(width=width, height=height)

    camera = CameraService(
        camera_index=camera_index,
        width=width,
        height=height,
        auto_resize=auto_resize,
        capture_device=capture_device,
    )

    try:
        camera.open()
    except CameraNotFoundError as exc:
        logger.error("Webcam hardware error: %s", exc)
        print("\n" + "=" * 60)
        print("CAMERA NOT FOUND / CANNOT OPEN WEBCAM")
        print("=" * 60)
        print(f"Details: {exc}")
        print("\nTroubleshooting tips:")
        print("  1. Verify the webcam is connected and recognized by Windows.")
        print("  2. Check Windows Settings -> Privacy -> Camera permissions.")
        print("  3. Try another index, e.g.: python scripts/demo_webcam_feed.py --camera-index 1")
        print("  4. Or run with simulated video: python scripts/demo_webcam_feed.py --mock")
        print("=" * 60 + "\n")
        return 1
    except CameraError as exc:
        logger.error("Failed to initialize camera: %s", exc)
        return 1

    window_title = "ReminisceCV - Live Camera Feed (Press 'q' or ESC to exit)"
    fps = 0.0
    frame_interval_start = time.time()
    frames_in_interval = 0

    print("\n" + "=" * 60)
    print("REMINISCECV WEBCAM FEED STARTED")
    print(f"Device Index:      {camera.camera_index}")
    print(f"Target Size:       {width}x{height}")
    print("Controls:          Press 'q' or ESC to exit cleanly")
    print("=" * 60 + "\n")

    try:
        while True:
            t0 = time.time()
            success, frame = camera.read_frame()

            if not success or frame is None:
                logger.warning("Frame acquisition returned empty or failed. Retrying...")
                time.sleep(0.01)
                continue

            frames_in_interval += 1
            elapsed_interval = time.time() - frame_interval_start
            if elapsed_interval >= 1.0:
                fps = frames_in_interval / elapsed_interval
                frames_in_interval = 0
                frame_interval_start = time.time()

            # Render overlay if not in headless mode
            if not headless:
                fh, fw = frame.shape[:2]
                # Status banner overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (320, 95), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

                cv2.putText(
                    frame,
                    f"ReminisceCV Camera Feed",
                    (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Res: {fw}x{fh} | FPS: {fps:.1f}",
                    (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (100, 240, 120),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Frame: #{camera.frame_count} | Press 'q' to Quit",
                    (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_title, frame)

                # 1ms waitKey gives OpenCV GUI event loop time to process & check keys
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q', 'Q', or ESC
                    logger.info("Quit key pressed ('%s'). Exiting clean...", chr(key) if key != 27 else "ESC")
                    break

                # Check if window was closed by the user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed by user. Exiting clean...")
                    break
            else:
                # In headless mode, log progress periodically
                if camera.frame_count % 30 == 0:
                    logger.info(
                        "Headless capture: frame #%d captured successfully (%dx%d)",
                        camera.frame_count,
                        frame.shape[1],
                        frame.shape[0],
                    )

            if max_frames > 0 and camera.frame_count >= max_frames:
                logger.info("Reached maximum frame limit of %d. Exiting...", max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received (Ctrl+C). Exiting clean...")
    finally:
        logger.info("Releasing camera hardware and destroying windows...")
        camera.release()
        if not headless:
            cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"Camera session ended cleanly. Total frames: {camera.frame_count}")
    print("=" * 60 + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReminisceCV — Live Webcam Feed Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=config.camera_index,
        help="Device index of the webcam to open.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config.frame_width,
        help="Target frame width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.frame_height,
        help="Target frame height in pixels.",
    )
    parser.add_argument(
        "--no-resize",
        dest="auto_resize",
        action="store_false",
        help="Disable automatic frame resizing to target width/height.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window display (useful for CI/testing).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to capture before auto-exiting (0 = unlimited).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic capture device instead of real hardware.",
    )

    args = parser.parse_args()
    code = run_camera_feed(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        auto_resize=args.auto_resize,
        headless=args.headless,
        max_frames=args.max_frames,
        use_mock=args.mock,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
