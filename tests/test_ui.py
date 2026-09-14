"""
tests/test_ui.py — Tests for OpenCV-Based UI and Memory Assistant Application
=============================================================================

Tests:
1. UIRenderer visual overlay elements on synthetic frames:
   - Header HUD and state badges (IDLE, ACQUIRING, STABLE).
   - Bounding boxes for faces and recognized objects.
   - Memory details card with metadata (title, giver, occasion, year, narrative).
   - Speech status indicators (MUTED, SPEAKING, COOLDOWN, READY).
   - Boundary and corner clipping robustness.
2. MemoryAssistantApp orchestration:
   - Keyboard handling: Q/ESC (quit), R (reset), M (mute/unmute).
   - Speech triggering on temporal stability.
   - Speech suppression under mute and cooldown.
   - Clean headless execution with mock devices.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from app.database.models import Memory
from app.memory.retrieval import StructuredMemoryResponse
from app.recognition.person_service import PersonRecognitionResult
from app.recognition.pipeline import (
    FrameRecognitionResult,
    PipelineTiming,
    RealTimeRecognitionPipeline,
)
from app.recognition.region_proposal import CandidateRegion
from app.recognition.service import RecognitionResult
from app.recognition.tracker import (
    DualRecognitionState,
    DualRecognitionTracker,
    RecognitionState,
    TemporalStatus,
    TrackedEntity,
)
from app.speech.cooldown import SpeechCooldownController
from app.speech.service import MockSpeechEngine, SpeechService
from app.ui.app import MemoryAssistantApp
from app.ui.renderer import (
    COLOR_AMBER,
    COLOR_BLUE,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    SpeechUIState,
    UIRenderer,
)
from app.vision.camera import CameraService
from app.vision.face import FaceBoundingBox


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """A standard 640x480 3-channel uint8 frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def renderer() -> UIRenderer:
    """Standard UI renderer."""
    return UIRenderer()


@pytest.fixture
def sample_face_result() -> PersonRecognitionResult:
    """Sample recognized person with bounding box."""
    return PersonRecognitionResult(
        matched=True,
        name="Sarah Connor",
        similarity=0.88,
        entity_id=1,
        entity_type="person",
        relationship="Daughter",
        bbox=FaceBoundingBox(x=100, y=100, width=120, height=140),
    )


@pytest.fixture
def sample_object_result() -> RecognitionResult:
    """Sample recognized object."""
    return RecognitionResult(
        entity_id=2,
        entity_type="object",
        name="Vintage Pocket Watch",
        similarity=0.91,
        matched=True,
    )


@pytest.fixture
def sample_memory_response() -> StructuredMemoryResponse:
    """Sample structured memory response."""
    return StructuredMemoryResponse(
        found=True,
        entity_id=2,
        entity_type="object",
        title="Vintage Pocket Watch",
        giver="Grandfather William",
        occasion="University Graduation",
        year="1975",
        narrative="William passed this down to celebrate graduating with honors in engineering.",
        is_active=True,
    )


# ── Renderer Unit Tests ───────────────────────────────────────────────────────

def test_renderer_initialization(renderer: UIRenderer) -> None:
    """UIRenderer initializes with expected font and settings."""
    assert renderer.font == cv2.FONT_HERSHEY_SIMPLEX
    assert renderer.show_fps is True


def test_renderer_empty_frame(renderer: UIRenderer) -> None:
    """Empty or None frames are returned unmodified."""
    assert renderer.render(None) is None
    empty = np.array([])
    assert renderer.render(empty).size == 0


def test_renderer_idle_frame(renderer: UIRenderer, synthetic_frame: np.ndarray) -> None:
    """Renderer handles idle state with no detection without error."""
    timing = PipelineTiming(fps=30.0, total_latency_ms=33.3)
    result = FrameRecognitionResult(
        frame_number=1,
        timestamp=time.time(),
        frame=synthetic_frame,
        timing=timing,
    )
    speech_state = SpeechUIState(is_muted=False, is_speaking=False)

    annotated = renderer.render(synthetic_frame, result, speech_state)

    assert annotated.shape == synthetic_frame.shape
    assert annotated.dtype == np.uint8
    # Canvas should have modified pixels from HUD and footer overlays
    assert not np.array_equal(annotated, synthetic_frame)


def test_renderer_draws_face_bounding_boxes(
    renderer: UIRenderer,
    synthetic_frame: np.ndarray,
    sample_face_result: PersonRecognitionResult,
) -> None:
    """Renderer draws bounding boxes and badges for matched and unmatched faces."""
    unmatched_face = PersonRecognitionResult(
        matched=False,
        name="unknown",
        similarity=0.22,
        bbox=FaceBoundingBox(x=350, y=120, width=90, height=110),
    )

    result = FrameRecognitionResult(
        frame_number=1,
        timestamp=time.time(),
        frame=synthetic_frame,
        detected_people=[sample_face_result, unmatched_face],
        timing=PipelineTiming(fps=25.0),
    )

    annotated = renderer.render(synthetic_frame, result)
    assert annotated.shape == synthetic_frame.shape
    # Check that pixels within the box perimeter were drawn
    assert np.any(annotated[100:240, 100:220] > 0)
    assert np.any(annotated[120:230, 350:440] > 0)


def test_renderer_draws_object_bounding_boxes(
    renderer: UIRenderer,
    synthetic_frame: np.ndarray,
    sample_object_result: RecognitionResult,
) -> None:
    """Renderer highlights candidate object region bounding boxes."""
    candidate = CandidateRegion(
        bbox=(200, 180, 140, 140),
        crop=np.zeros((140, 140, 3), dtype=np.uint8),
        region_type="center_focus",
        area=140 * 140,
    )
    result = FrameRecognitionResult(
        frame_number=1,
        timestamp=time.time(),
        frame=synthetic_frame,
        recognized_objects=[sample_object_result],
        candidate_regions=[candidate],
        timing=PipelineTiming(fps=28.0),
    )

    annotated = renderer.render(synthetic_frame, result)
    assert annotated.shape == synthetic_frame.shape
    assert np.any(annotated[180:320, 200:340] > 0)


def test_renderer_draws_memory_card(
    renderer: UIRenderer,
    synthetic_frame: np.ndarray,
    sample_memory_response: StructuredMemoryResponse,
) -> None:
    """Renderer displays memory card with title, giver, occasion, year, narrative."""
    tracker_state = DualRecognitionState(
        object_state=RecognitionState(
            status=TemporalStatus.STABLE,
            candidate=TrackedEntity(
                entity_id=2,
                entity_type="object",
                name="Vintage Pocket Watch",
                similarity=0.92,
            ),
            is_stable=True,
            elapsed_duration_sec=2.0,
            consecutive_frames=15,
        ),
        person_state=RecognitionState(status=TemporalStatus.IDLE),
    )

    result = FrameRecognitionResult(
        frame_number=45,
        timestamp=time.time(),
        frame=synthetic_frame,
        tracker_state=tracker_state,
        stable_memory=sample_memory_response,
        timing=PipelineTiming(fps=30.0),
    )

    annotated = renderer.render(synthetic_frame, result)
    assert annotated.shape == synthetic_frame.shape
    # Bottom memory card area should be rendered
    card_area = annotated[340:450, 16:624]
    assert np.any(card_area > 0)


def test_renderer_speech_status_variations(
    renderer: UIRenderer, synthetic_frame: np.ndarray
) -> None:
    """Renderer displays appropriate badges for muted, speaking, and cooldown states."""
    # 1. Muted state
    st_muted = SpeechUIState(is_muted=True)
    out_muted = renderer.render(synthetic_frame, speech_state=st_muted)
    assert np.any(out_muted[:54, 300:500] > 0)

    # 2. Speaking state
    st_speaking = SpeechUIState(is_speaking=True)
    out_speaking = renderer.render(synthetic_frame, speech_state=st_speaking)
    assert np.any(out_speaking[:54, 300:500] > 0)

    # 3. Cooldown state
    st_cooldown = SpeechUIState(cooldown_remaining=14.5)
    out_cooldown = renderer.render(synthetic_frame, speech_state=st_cooldown)
    assert np.any(out_cooldown[:54, 300:500] > 0)


def test_renderer_boundary_clipping(renderer: UIRenderer) -> None:
    """Renderer gracefully clips bounding boxes at image borders."""
    tiny_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Box extending beyond frame borders
    renderer._draw_corner_box(tiny_frame, x=-20, y=-10, w=150, h=140, color=(0, 255, 0))
    renderer._draw_alpha_rect(tiny_frame, x=-10, y=-5, w=200, h=50, color=(10, 10, 10))
    renderer._draw_badge(tiny_frame, "TEST", x=85, y=85, bg_color=(255, 0, 0))

    assert tiny_frame.shape == (100, 100, 3)


# ── MemoryAssistantApp Unit Tests ─────────────────────────────────────────────

@pytest.fixture
def mock_app() -> MemoryAssistantApp:
    """App fixture with mock pipeline, mock speech, and cooldown controller."""
    mock_pipeline = MagicMock(spec=RealTimeRecognitionPipeline)
    mock_pipeline.is_running = True
    mock_pipeline.tracker = MagicMock(spec=DualRecognitionTracker)

    speech_engine = MockSpeechEngine()
    speech_service = SpeechService(engine=speech_engine, auto_start=True)
    cooldown = SpeechCooldownController(default_cooldown=30.0)
    renderer = UIRenderer()

    app = MemoryAssistantApp(
        pipeline=mock_pipeline,
        speech_service=speech_service,
        cooldown_controller=cooldown,
        renderer=renderer,
    )
    return app


def test_app_keyboard_controls(mock_app: MemoryAssistantApp) -> None:
    """Test Q (quit), R (reset), and M (mute) keyboard handling."""
    mock_app.is_running = True

    # 1. 'M' toggles mute
    assert mock_app.is_muted is False
    assert mock_app.handle_key(ord("m")) is True
    assert mock_app.is_muted is True
    assert mock_app.handle_key(ord("M")) is True
    assert mock_app.is_muted is False

    # 2. 'R' resets recognition tracker
    assert mock_app.handle_key(ord("r")) is True
    mock_app.pipeline.tracker.reset.assert_called()

    # 3. 'Q' requests clean quit
    assert mock_app.handle_key(ord("q")) is False
    assert mock_app.is_running is False

    # 4. ESC (27) requests clean quit
    mock_app.is_running = True
    assert mock_app.handle_key(27) is False
    assert mock_app.is_running is False


def test_app_speech_trigger_on_became_stable(
    mock_app: MemoryAssistantApp,
    sample_memory_response: StructuredMemoryResponse,
    synthetic_frame: np.ndarray,
) -> None:
    """App announces memory via SpeechService when an entity achieves stability."""
    tracker_state = DualRecognitionState(
        object_state=RecognitionState(
            status=TemporalStatus.STABLE,
            candidate=TrackedEntity(
                entity_id=2,
                entity_type="object",
                name="Vintage Pocket Watch",
                similarity=0.92,
            ),
            is_stable=True,
            became_stable=True,
        ),
        person_state=RecognitionState(status=TemporalStatus.IDLE),
    )

    result = FrameRecognitionResult(
        frame_number=1,
        timestamp=time.time(),
        frame=synthetic_frame,
        tracker_state=tracker_state,
        stable_memory=sample_memory_response,
    )
    mock_app.pipeline.process_frame.return_value = result

    # Execute one step
    res, annotated = mock_app.process_step(frame=synthetic_frame, headless=True)

    assert res.became_stable is True
    assert mock_app.last_announced_entity_id == 2
    assert "Vintage Pocket Watch" in mock_app.last_spoken_text
    # Cooldown should now be active
    assert mock_app.cooldown.can_announce(entity_id=2, entity_type="object") is False


def test_app_speech_suppressed_when_muted(
    mock_app: MemoryAssistantApp,
    sample_memory_response: StructuredMemoryResponse,
    synthetic_frame: np.ndarray,
) -> None:
    """Muted speech prevents queuing any audio utterance."""
    mock_app.is_muted = True

    tracker_state = DualRecognitionState(
        object_state=RecognitionState(
            status=TemporalStatus.STABLE,
            candidate=TrackedEntity(
                entity_id=2,
                entity_type="object",
                name="Vintage Pocket Watch",
                similarity=0.92,
            ),
            is_stable=True,
            became_stable=True,
        ),
        person_state=RecognitionState(status=TemporalStatus.IDLE),
    )

    result = FrameRecognitionResult(
        frame_number=1,
        timestamp=time.time(),
        frame=synthetic_frame,
        tracker_state=tracker_state,
        stable_memory=sample_memory_response,
    )
    mock_app.pipeline.process_frame.return_value = result

    res, annotated = mock_app.process_step(frame=synthetic_frame, headless=True)

    # Utterance should be suppressed
    assert mock_app.last_announced_entity_id is None
    assert mock_app.last_spoken_text == ""


def test_app_speech_suppressed_when_in_cooldown(
    mock_app: MemoryAssistantApp,
    sample_memory_response: StructuredMemoryResponse,
    synthetic_frame: np.ndarray,
) -> None:
    """Pre-existing cooldown prevents duplicate speech announcements."""
    # Pre-record announcement so entity is on cooldown
    mock_app.cooldown.record_announcement(entity_id=2, entity_type="object")

    tracker_state = DualRecognitionState(
        object_state=RecognitionState(
            status=TemporalStatus.STABLE,
            candidate=TrackedEntity(
                entity_id=2,
                entity_type="object",
                name="Vintage Pocket Watch",
                similarity=0.92,
            ),
            is_stable=True,
            became_stable=True,
        ),
        person_state=RecognitionState(status=TemporalStatus.IDLE),
    )

    result = FrameRecognitionResult(
        frame_number=1,
        timestamp=time.time(),
        frame=synthetic_frame,
        tracker_state=tracker_state,
        stable_memory=sample_memory_response,
    )
    mock_app.pipeline.process_frame.return_value = result

    res, annotated = mock_app.process_step(frame=synthetic_frame, headless=True)

    # Should be suppressed by cooldown
    assert mock_app.last_spoken_text == ""


def test_app_run_max_frames(
    mock_app: MemoryAssistantApp, synthetic_frame: np.ndarray
) -> None:
    """App executes specified max_frames and exits cleanly."""
    result = FrameRecognitionResult(
        frame_number=1,
        timestamp=time.time(),
        frame=synthetic_frame,
        timing=PipelineTiming(fps=30.0),
    )
    mock_app.pipeline.process_frame.return_value = result

    mock_app.run(max_frames=5, headless=True)

    assert mock_app.frame_count == 5
    assert mock_app.is_running is False


def test_app_lifecycle_context_manager(mock_app: MemoryAssistantApp) -> None:
    """App supports context management with clean start and stop."""
    with mock_app as app:
        assert app.is_running is True

    assert mock_app.is_running is False
