"""
scripts/demo_realtime_recognition.py — Live Webcam Real-Time Recognition Demo
=============================================================================

Executes ReminisceCV's modular real-time recognition pipeline on live webcam frames:
1. Captures frames in real time using CameraService.
2. Extracts candidate visual regions (center focus & contours).
3. Detects and recognizes familiar faces (with green/amber bounding box overlays).
4. Matches candidate object crops against registered personal memories.
5. Displays a live performance telemetry HUD (latencies and real-time FPS).
6. Shuts down cleanly on pressing 'q' or ESC.

Usage
-----
Live interactive camera feed:
    python scripts/demo_realtime_recognition.py

Custom camera and resolution:
    python scripts/demo_realtime_recognition.py --camera-index 0 --width 640 --height 480

Headless verification (e.g. CI / tests):
    python scripts/demo_realtime_recognition.py --mock --headless --max-frames 20
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
from app.recognition import (
    FrameRecognitionResult,
    ObjectRecognitionService,
    PersonRecognitionService,
    RealTimeRecognitionPipeline,
)
from app.vision import (
    CameraError,
    CameraNotFoundError,
    CameraService,
    MockFaceEmbeddingModel,
    MockVisionEmbeddingModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo_realtime_recognition")


class SyntheticVideoCapture:
    """Mock capture device generating animated test frames for headless/demo testing."""

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
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :, 0] = np.linspace(35, 75, self.height, dtype=np.uint8)[:, None]
        frame[:, :, 1] = np.linspace(45, 95, self.height, dtype=np.uint8)[:, None]
        frame[:, :, 2] = np.linspace(55, 115, self.height, dtype=np.uint8)[:, None]

        # Moving shape (simulated object)
        cx = int((np.sin(self._frame_idx * 0.06) * 0.25 + 0.5) * self.width)
        cy = int((np.cos(self._frame_idx * 0.06) * 0.2 + 0.5) * self.height)
        cv2.circle(frame, (cx, cy), 45, (80, 210, 110), -1)

        time.sleep(0.02)  # Emulate real-time camera pace
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


def render_visual_overlays(
    frame: np.ndarray, result: FrameRecognitionResult
) -> np.ndarray:
    """Render bounding boxes, recognition banners, and performance telemetry on frame."""
    annotated = frame.copy()
    fh, fw = annotated.shape[:2]

    # 1. Draw detected face bounding boxes
    for person in result.detected_people:
        if person.bbox:
            x, y, w, h = (
                person.bbox.x,
                person.bbox.y,
                person.bbox.width,
                person.bbox.height,
            )
            color = (50, 220, 80) if person.matched else (40, 140, 240)  # Green if matched, Amber if unknown
            label = f"{person.name} ({person.similarity:.2f})"

            # Box & corner highlights
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(
                annotated, (x, max(0, y - 24)), (x + len(label) * 9 + 10, max(0, y)), color, -1
            )
            cv2.putText(
                annotated,
                label,
                (x + 5, max(16, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    # 2. Draw recognized object banner at bottom if top object detected
    top_obj = result.top_object
    if top_obj and top_obj.matched:
        banner = annotated.copy()
        cv2.rectangle(banner, (0, fh - 65), (fw, fh), (20, 20, 20), -1)
        cv2.addWeighted(banner, 0.7, annotated, 0.3, 0, annotated)

        title = f"Recognized: {top_obj.name} (similarity: {top_obj.similarity:.2f})"
        cv2.putText(
            annotated,
            title,
            (16, fh - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )
        if top_obj.memory and top_obj.memory.giver_name:
            sub = f"From: {top_obj.memory.giver_name} | Occasion: {top_obj.memory.occasion or 'N/A'}"
            cv2.putText(
                annotated,
                sub,
                (16, fh - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

    # 3. Performance Telemetry HUD (Top-Left)
    hud = annotated.copy()
    cv2.rectangle(hud, (8, 8), (280, 130), (15, 15, 15), -1)
    cv2.addWeighted(hud, 0.65, annotated, 0.35, 0, annotated)

    t = result.timing
    cv2.putText(
        annotated,
        f"ReminisceCV Live Pipeline",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"FPS: {t.fps:.1f} | Total: {t.total_latency_ms:.1f} ms",
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 240, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Face Latency:   {t.face_latency_ms:.1f} ms",
        (18, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Object Latency: {t.object_latency_ms:.1f} ms",
        (18, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Frame #{result.frame_number} | Press 'q' to exit",
        (18, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    return annotated


def run_pipeline_demo(
    camera_index: int,
    width: int,
    height: int,
    enable_objects: bool = True,
    enable_faces: bool = True,
    headless: bool = False,
    max_frames: int = 0,
    use_mock: bool = False,
) -> int:
    """Run real-time webcam recognition pipeline.

    Returns
    -------
    int
        Exit status (0 on clean exit).
    """
    logger.info("Setting up ReminisceCV Real-Time Recognition Pipeline...")

    # Hardware camera or mock capture device
    if use_mock:
        logger.info("Using SyntheticVideoCapture device.")
        capture_device = SyntheticVideoCapture(width=width, height=height)
        camera = CameraService(
            camera_index=camera_index,
            width=width,
            height=height,
            capture_device=capture_device,
        )
        # In mock mode, use lightweight mock models to run instantly without weight downloads
        vision_model = MockVisionEmbeddingModel()
        face_engine = MockFaceEmbeddingModel()
        obj_service = ObjectRecognitionService(vision_model=vision_model)
        person_service = PersonRecognitionService(face_engine=face_engine)
    else:
        camera = CameraService(
            camera_index=camera_index,
            width=width,
            height=height,
        )
        obj_service = ObjectRecognitionService()
        person_service = PersonRecognitionService()

    pipeline = RealTimeRecognitionPipeline(
        camera_service=camera,
        object_service=obj_service,
        person_service=person_service,
        enable_objects=enable_objects,
        enable_faces=enable_faces,
        auto_warmup=True,
    )

    try:
        pipeline.start()
    except CameraNotFoundError as exc:
        logger.error("Camera hardware not available: %s", exc)
        print("\n" + "=" * 60)
        print("CAMERA NOT FOUND")
        print(f"Details: {exc}")
        print("\nTip: Run with simulated video: python scripts/demo_realtime_recognition.py --mock")
        print("=" * 60 + "\n")
        return 1
    except Exception as exc:
        logger.error("Failed to start pipeline: %s", exc)
        return 1

    window_title = "ReminisceCV — Real-Time Recognition (Press 'q' or ESC to exit)"
    frame_count = 0

    print("\n" + "=" * 60)
    print("REMINISCECV REAL-TIME RECOGNITION PIPELINE ACTIVE")
    print(f"Camera Index:     {camera_index} ({width}x{height})")
    print(f"Object Matching:  {'Enabled' if enable_objects else 'Disabled'}")
    print(f"Face Matching:    {'Enabled' if enable_faces else 'Disabled'}")
    print("Controls:         Press 'q' or ESC to quit cleanly")
    print("=" * 60 + "\n")

    try:
        for result in pipeline.stream():
            frame_count += 1

            if not headless:
                display_frame = render_visual_overlays(result.frame, result)
                cv2.imshow(window_title, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    logger.info("Quit requested by user key press.")
                    break

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed by user.")
                    break
            else:
                if frame_count % 10 == 0:
                    t = result.timing
                    logger.info(
                        "Frame #%d: total=%.1f ms (face=%.1f ms, obj=%.1f ms) | FPS=%.1f | matches=%s",
                        result.frame_number,
                        t.total_latency_ms,
                        t.face_latency_ms,
                        t.object_latency_ms,
                        t.fps,
                        result.has_matches,
                    )

            if max_frames > 0 and frame_count >= max_frames:
                logger.info("Reached frame limit of %d. Exiting...", max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Shutting down pipeline cleanly...")
        pipeline.stop()
        if not headless:
            cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"Pipeline stopped cleanly. Total frames processed: {frame_count}")
    print("=" * 60 + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReminisceCV — Real-Time Webcam Recognition Pipeline Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=config.camera_index,
        help="Camera device index to open.",
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
        "--no-faces",
        dest="enable_faces",
        action="store_false",
        help="Disable familiar face recognition stage.",
    )
    parser.add_argument(
        "--no-objects",
        dest="enable_objects",
        action="store_false",
        help="Disable personal object recognition stage.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window (for benchmarking or CI).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process before auto-exit (0 = run indefinitely).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic video feed and mock models.",
    )

    args = parser.parse_args()
    code = run_pipeline_demo(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        enable_objects=args.enable_objects,
        enable_faces=args.enable_faces,
        headless=args.headless,
        max_frames=args.max_frames,
        use_mock=args.mock,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
