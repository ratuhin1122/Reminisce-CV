"""
tests.test_realtime_pipeline — Unit Tests for Real-Time Webcam Recognition Pipeline
===================================================================================

Tests cover:
    - Candidate visual region extraction (full frame, center focus, contour proposals)
    - Pipeline warmup and upfront model loading (zero model reloading in frame loop)
    - Single-frame processing and structured FrameRecognitionResult output
    - Timing metrics verification (capture time, stage latencies, FPS)
    - Modularity toggles (disabling face stage or object stage independently)
    - Recognition matching on live video frames with injected gallery embeddings
    - Streaming generator and clean shutdown/release lifecycle
"""

import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.config import config
from app.database.models import Memory
from app.recognition import (
    CandidateRegion,
    CandidateRegionExtractor,
    FrameRecognitionResult,
    ObjectRecognitionService,
    PersonRecognitionResult,
    PersonRecognitionService,
    PipelineTiming,
    RealTimeRecognitionPipeline,
    RecognitionResult,
)
from app.vision import (
    CameraService,
    FaceBoundingBox,
    MockFaceEmbeddingModel,
    MockVisionEmbeddingModel,
)


# ── Mock Fixtures & Helpers ───────────────────────────────────────────────────


class MockCaptureDevice:
    """Simulated camera capture device."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._opened = True
        self.released = False
        self.frame_count = 0

    def isOpened(self) -> bool:
        return self._opened and not self.released

    def read(self) -> tuple[bool, np.ndarray]:
        if not self.isOpened():
            return False, np.array([])
        self.frame_count += 1
        # Create a sample frame with a distinct center shape
        frame = np.full((self.height, self.width, 3), 60, dtype=np.uint8)
        cv2.circle(frame, (self.width // 2, self.height // 2), 60, (200, 200, 200), -1)
        return True, frame

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def release(self) -> None:
        self.released = True
        self._opened = False


@pytest.fixture
def mock_camera() -> CameraService:
    dev = MockCaptureDevice(width=640, height=480)
    return CameraService(camera_index=0, width=640, height=480, capture_device=dev)


@pytest.fixture
def mock_vision_model() -> MockVisionEmbeddingModel:
    return MockVisionEmbeddingModel(embedding_dim=512, auto_load=True)


@pytest.fixture
def mock_face_engine() -> MockFaceEmbeddingModel:
    return MockFaceEmbeddingModel(embedding_dim=512, auto_load=True)


@pytest.fixture
def mock_pipeline(
    mock_camera: CameraService,
    mock_vision_model: MockVisionEmbeddingModel,
    mock_face_engine: MockFaceEmbeddingModel,
) -> RealTimeRecognitionPipeline:
    obj_service = ObjectRecognitionService(vision_model=mock_vision_model)
    person_service = PersonRecognitionService(face_engine=mock_face_engine)
    return RealTimeRecognitionPipeline(
        camera_service=mock_camera,
        object_service=obj_service,
        person_service=person_service,
        auto_warmup=True,
    )


# ── Region Extractor Tests ────────────────────────────────────────────────────


class TestCandidateRegionExtractor:
    """Tests for visual region proposal generation."""

    def test_empty_frame_returns_empty_proposals(self) -> None:
        extractor = CandidateRegionExtractor()
        assert extractor.extract(np.array([])) == []

    def test_full_frame_and_center_crop_extracted(self) -> None:
        extractor = CandidateRegionExtractor(
            include_full_frame=True,
            include_center_crop=True,
            include_contours=False,
            center_crop_ratio=0.5,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = extractor.extract(frame)

        assert len(regions) == 2
        types = [r.region_type for r in regions]
        assert "full_frame" in types
        assert "center_focus" in types

        full = next(r for r in regions if r.region_type == "full_frame")
        assert full.bbox == (0, 0, 640, 480)
        assert full.crop.shape == (480, 640, 3)

        center = next(r for r in regions if r.region_type == "center_focus")
        assert center.bbox == (160, 120, 320, 240)
        assert center.crop.shape == (240, 320, 3)

    def test_salient_contour_proposals(self) -> None:
        extractor = CandidateRegionExtractor(
            include_full_frame=False,
            include_center_crop=False,
            include_contours=True,
            min_contour_area=500,
            max_contour_proposals=2,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a prominent white rectangle
        cv2.rectangle(frame, (100, 100), (250, 250), (255, 255, 255), -1)

        regions = extractor.extract(frame)
        assert len(regions) >= 1
        assert regions[0].region_type == "salient_contour"
        assert regions[0].area > 500


# ── Pipeline Lifecycle & Warmup Tests ─────────────────────────────────────────


class TestPipelineWarmupAndLifecycle:
    """Tests verifying pre-loading and lifecycle management."""

    def test_warmup_preloads_models_and_refreshes_galleries(
        self,
        mock_camera: CameraService,
        mock_vision_model: MockVisionEmbeddingModel,
        mock_face_engine: MockFaceEmbeddingModel,
    ) -> None:
        # Start with unloaded models
        mock_vision_model._is_loaded = False
        mock_face_engine._is_loaded = False

        obj_service = ObjectRecognitionService(vision_model=mock_vision_model)
        person_service = PersonRecognitionService(face_engine=mock_face_engine)

        pipeline = RealTimeRecognitionPipeline(
            camera_service=mock_camera,
            object_service=obj_service,
            person_service=person_service,
            auto_warmup=False,
        )
        assert pipeline._models_warmed_up is False

        # Run warmup
        pipeline.warmup()
        assert pipeline._models_warmed_up is True
        assert mock_vision_model.is_loaded is True
        assert mock_face_engine.is_loaded is True

    def test_models_not_reloaded_in_frame_loop(
        self, mock_pipeline: RealTimeRecognitionPipeline
    ) -> None:
        # Wrap load_model methods with mocks to count any subsequent invocations
        mock_pipeline.object_service.vision_model.load_model = MagicMock(
            wraps=mock_pipeline.object_service.vision_model.load_model
        )
        mock_pipeline.person_service.face_engine.load_model = MagicMock(
            wraps=mock_pipeline.person_service.face_engine.load_model
        )

        # Process multiple frames
        for _ in range(5):
            mock_pipeline.process_frame()

        # load_model must NOT be invoked inside the frame processing loop
        mock_pipeline.object_service.vision_model.load_model.assert_not_called()
        mock_pipeline.person_service.face_engine.load_model.assert_not_called()

    def test_context_manager_starts_and_stops_cleanly(
        self,
        mock_camera: CameraService,
        mock_vision_model: MockVisionEmbeddingModel,
        mock_face_engine: MockFaceEmbeddingModel,
    ) -> None:
        obj_service = ObjectRecognitionService(vision_model=mock_vision_model)
        person_service = PersonRecognitionService(face_engine=mock_face_engine)

        with RealTimeRecognitionPipeline(
            camera_service=mock_camera,
            object_service=obj_service,
            person_service=person_service,
        ) as pipeline:
            assert pipeline._is_running is True
            assert mock_camera.is_opened is True

        assert pipeline._is_running is False
        assert mock_camera.is_opened is False


# ── Frame Processing & Structured Output Tests ────────────────────────────────


class TestPipelineFrameProcessing:
    """Tests for single frame processing and structured results."""

    def test_process_frame_from_camera(
        self, mock_pipeline: RealTimeRecognitionPipeline
    ) -> None:
        result = mock_pipeline.process_frame()

        assert isinstance(result, FrameRecognitionResult)
        assert result.frame_number == 1
        assert result.timestamp > 0
        assert result.frame.shape == (480, 640, 3)
        assert isinstance(result.timing, PipelineTiming)
        assert result.timing.capture_time_ms >= 0.0
        assert result.timing.total_latency_ms > 0.0

    def test_process_external_frame(
        self, mock_pipeline: RealTimeRecognitionPipeline
    ) -> None:
        test_frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        result = mock_pipeline.process_frame(frame=test_frame, frame_number=42)

        assert result.frame_number == 42
        assert result.frame.shape == (240, 320, 3)
        assert result.timing.capture_time_ms == 0.0  # Externally provided

    def test_prepare_frame_ensures_uint8_contiguous(
        self, mock_pipeline: RealTimeRecognitionPipeline
    ) -> None:
        float_frame = np.zeros((100, 100, 3), dtype=np.float32)
        prepared = mock_pipeline.prepare_frame(float_frame)
        assert prepared.dtype == np.uint8
        assert prepared.flags.c_contiguous

    def test_prepare_empty_frame_raises(
        self, mock_pipeline: RealTimeRecognitionPipeline
    ) -> None:
        with pytest.raises(ValueError, match="empty or None"):
            mock_pipeline.prepare_frame(np.array([]))


# ── Modularity & Recognition Matching Tests ───────────────────────────────────


class TestPipelineModularityAndMatching:
    """Tests verifying independent feature toggles and matching logic."""

    def test_disable_faces_skips_face_stage(
        self,
        mock_camera: CameraService,
        mock_vision_model: MockVisionEmbeddingModel,
        mock_face_engine: MockFaceEmbeddingModel,
    ) -> None:
        obj_service = ObjectRecognitionService(vision_model=mock_vision_model)
        person_service = PersonRecognitionService(face_engine=mock_face_engine)

        pipeline = RealTimeRecognitionPipeline(
            camera_service=mock_camera,
            object_service=obj_service,
            person_service=person_service,
            enable_faces=False,
            enable_objects=True,
        )

        result = pipeline.process_frame()
        assert result.detected_people == []
        assert result.timing.face_latency_ms == 0.0
        assert len(result.candidate_regions) > 0

    def test_disable_objects_skips_object_stage(
        self,
        mock_camera: CameraService,
        mock_vision_model: MockVisionEmbeddingModel,
        mock_face_engine: MockFaceEmbeddingModel,
    ) -> None:
        obj_service = ObjectRecognitionService(vision_model=mock_vision_model)
        person_service = PersonRecognitionService(face_engine=mock_face_engine)

        pipeline = RealTimeRecognitionPipeline(
            camera_service=mock_camera,
            object_service=obj_service,
            person_service=person_service,
            enable_faces=True,
            enable_objects=False,
        )

        result = pipeline.process_frame()
        assert result.recognized_objects == []
        assert result.candidate_regions == []
        assert result.timing.object_latency_ms == 0.0

    def test_positive_face_match_reflected_in_result(
        self,
        mock_pipeline: RealTimeRecognitionPipeline,
        mock_face_engine: MockFaceEmbeddingModel,
    ) -> None:
        # Configure mock face detection
        mock_face_engine.set_mock_bboxes([FaceBoundingBox(50, 50, 80, 80)])

        # Set face engine to return a known embedding
        known_emb = np.zeros(512, dtype=np.float32)
        known_emb[0] = 1.0
        mock_face_engine.encode_face = MagicMock(return_value=known_emb)

        # Inject matching gallery embedding
        mock_pipeline.person_service.set_in_memory_gallery([known_emb], [101])

        # Mock database memory return
        mock_pipeline.person_service._mem_service.get_memory = MagicMock(
            return_value=Memory(id=101, name="Alice", giver_name="Daughter")
        )

        result = mock_pipeline.process_frame()
        assert len(result.detected_people) == 1
        person = result.detected_people[0]
        assert person.matched is True
        assert person.name == "Alice"
        assert result.has_matches is True
        assert result.top_person is not None
        assert result.top_person.name == "Alice"

    def test_positive_object_match_reflected_in_result(
        self,
        mock_pipeline: RealTimeRecognitionPipeline,
        mock_vision_model: MockVisionEmbeddingModel,
    ) -> None:
        known_emb = np.zeros(512, dtype=np.float32)
        known_emb[5] = 1.0
        mock_vision_model.encode_image = MagicMock(return_value=known_emb)

        # Inject matching object in gallery
        mock_pipeline.object_service.set_in_memory_gallery([known_emb], [202])
        mock_pipeline.object_service._mem_service.get_memory = MagicMock(
            return_value=Memory(id=202, name="Vintage Watch", giver_name="Grandfather")
        )

        result = mock_pipeline.process_frame()
        assert result.has_matches is True
        assert result.top_object is not None
        assert result.top_object.name == "Vintage Watch"
        assert result.top_object.similarity >= 0.9



# ── Streaming Tests ───────────────────────────────────────────────────────────


class TestPipelineStreaming:
    """Tests for continuous frame streaming generator."""

    def test_stream_generator_yields_frames(
        self, mock_pipeline: RealTimeRecognitionPipeline
    ) -> None:
        mock_pipeline.start()
        frames_received = 0

        for res in mock_pipeline.stream():
            frames_received += 1
            assert isinstance(res, FrameRecognitionResult)
            if frames_received >= 3:
                mock_pipeline.stop()
                break

        assert frames_received == 3
        assert mock_pipeline._is_running is False
