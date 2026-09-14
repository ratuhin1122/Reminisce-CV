"""
scripts/run_assistant.py — ReminisceCV Real-Time Memory Assistant Launcher
==========================================================================

Runs the OpenCV-based real-time memory assistant interface:
- Live camera video capture with bounding boxes, state badges, and memory card overlays.
- Local AI inference for personal objects and familiar faces.
- Non-blocking text-to-speech audio narration with per-memory cooldown.
- Keyboard controls:
    [Q] or ESC -> Quit cleanly
    [R]        -> Reset recognition tracking state
    [M]        -> Mute / unmute audio narration

Usage
-----
Interactive live camera feed:
    python scripts/run_assistant.py

Select specific camera device and resolution:
    python scripts/run_assistant.py --camera 0 --width 1280 --height 720

Start with speech muted:
    python scripts/run_assistant.py --mute

Headless verification (CI / unit testing):
    python scripts/run_assistant.py --mock --headless --max-frames 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.memory.retrieval import MemoryRetrievalService
from app.recognition.person_service import PersonRecognitionService
from app.recognition.pipeline import RealTimeRecognitionPipeline
from app.recognition.service import ObjectRecognitionService
from app.recognition.tracker import DualRecognitionTracker
from app.speech.cooldown import SpeechCooldownController
from app.speech.service import MockSpeechEngine, SpeechService
from app.ui.app import MemoryAssistantApp
from app.ui.renderer import UIRenderer
from app.vision import CameraService, MockFaceEmbeddingModel, MockVisionEmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reminisce_assistant")


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

        # Moving shape (simulated personal object)
        cx = int((np.sin(self._frame_idx * 0.08) * 0.25 + 0.5) * self.width)
        cy = int((np.cos(self._frame_idx * 0.08) * 0.2 + 0.5) * self.height)
        cv2.circle(frame, (cx, cy), 45, (80, 210, 110), -1)

        time.sleep(0.02)  # Emulate camera frame interval
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


def build_app(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    use_mock: bool = False,
    start_muted: bool = False,
    stability_duration: float = 1.5,
    cooldown_seconds: float = 30.0,
) -> MemoryAssistantApp:
    """Instantiate and configure the complete MemoryAssistantApp."""
    if use_mock:
        logger.info("Initializing ReminisceCV assistant in MOCK mode...")
        mock_cap = SyntheticVideoCapture(width=width, height=height)
        camera = CameraService(
            camera_index=camera_index,
            width=width,
            height=height,
            capture_device=mock_cap,
        )
        obj_service = ObjectRecognitionService(
            vision_model=MockVisionEmbeddingModel(embedding_dim=512, auto_load=True)
        )
        person_service = PersonRecognitionService(
            face_engine=MockFaceEmbeddingModel(embedding_dim=512, auto_load=True)
        )
        speech_engine = MockSpeechEngine()
        speech_service = SpeechService(engine=speech_engine)
    else:
        logger.info("Initializing ReminisceCV assistant with live camera hardware...")
        camera = CameraService(
            camera_index=camera_index,
            width=width,
            height=height,
        )
        obj_service = ObjectRecognitionService()
        person_service = PersonRecognitionService()
        speech_service = SpeechService()

    tracker = DualRecognitionTracker(
        object_stability_sec=stability_duration,
        person_stability_sec=stability_duration,
    )
    cooldown = SpeechCooldownController(default_cooldown=cooldown_seconds)
    retrieval = MemoryRetrievalService()

    pipeline = RealTimeRecognitionPipeline(
        camera_service=camera,
        object_service=obj_service,
        person_service=person_service,
        tracker=tracker,
        memory_retrieval=retrieval,
        auto_warmup=not use_mock,
    )

    renderer = UIRenderer()

    app = MemoryAssistantApp(
        pipeline=pipeline,
        speech_service=speech_service,
        cooldown_controller=cooldown,
        renderer=renderer,
        start_muted=start_muted,
    )
    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ReminisceCV Real-Time OpenCV Memory Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config.frame_width,
        help="Capture frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.frame_height,
        help="Capture frame height",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Start with speech narration muted",
    )
    parser.add_argument(
        "--stability",
        type=float,
        default=1.5,
        help="Required observation duration (seconds) before entity recognition stabilizes",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=30.0,
        help="Minimum duration (seconds) between repeated announcements of the same memory",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic video feed and lightweight mock AI models",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening an OpenCV GUI window (for tests/CI)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run indefinitely)",
    )

    args = parser.parse_args()

    app = build_app(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        use_mock=args.mock,
        start_muted=args.mute,
        stability_duration=args.stability,
        cooldown_seconds=args.cooldown,
    )

    try:
        app.run(
            max_frames=args.max_frames if args.max_frames > 0 else None,
            headless=args.headless,
        )
    except Exception as exc:
        logger.error("Assistant execution encountered an error: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
