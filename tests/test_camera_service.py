"""
tests.test_camera_service — Unit Tests for Camera & Webcam Capture Layer
========================================================================

Verifies the independent webcam capture layer:
- Default initialization using application config
- Context manager protocol (__enter__, __exit__)
- Camera availability validation
- Resolution configuration and automatic frame resizing
- Structured frame output (CameraFrame dataclass)
- Robust error handling (CameraNotFoundError, CameraError, FrameCaptureError)
- Clean, leak-free, idempotent shutdown and release
- Total independence from AI/recognition modules
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.config import config
from app.vision.camera import (
    CameraError,
    CameraFrame,
    CameraNotFoundError,
    CameraService,
    FrameCaptureError,
    WebcamService,
)


# ── Mock Capture Devices ──────────────────────────────────────────────────────


class MockCaptureSuccess:
    """Simulates a functioning OpenCV VideoCapture device."""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self._opened = True
        self.released = False

    def isOpened(self) -> bool:
        return self._opened and not self.released

    def read(self) -> tuple[bool, np.ndarray]:
        if not self.isOpened():
            return False, np.array([])
        frame = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
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


class MockCaptureFailOpen:
    """Simulates a camera index that fails to open."""

    def isOpened(self) -> bool:
        return False

    def read(self) -> tuple[bool, np.ndarray]:
        return False, np.array([])

    def set(self, prop: int, value: float) -> bool:
        return False

    def release(self) -> None:
        pass


class MockCaptureYieldsEmpty:
    """Simulates a camera that reports opened but yields no frames."""

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return False, None

    def get(self, prop: int) -> float:
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def release(self) -> None:
        pass


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCameraServiceInitialization:
    """Tests for initial state, configuration defaults, and aliases."""

    def test_default_initialization_matches_config(self) -> None:
        camera = CameraService()
        assert camera.camera_index == config.camera_index
        assert camera.target_size == (config.frame_width, config.frame_height)
        assert camera.is_opened is False
        assert camera.frame_count == 0

    def test_custom_parameters_override_defaults(self) -> None:
        camera = CameraService(camera_index=2, width=1920, height=1080, auto_resize=False)
        assert camera.camera_index == 2
        assert camera.target_size == (1920, 1080)
        assert camera.is_opened is False

    def test_webcam_service_is_alias_of_camera_service(self) -> None:
        assert WebcamService is CameraService
        ws = WebcamService(camera_index=1, width=320, height=240)
        assert isinstance(ws, CameraService)
        assert ws.camera_index == 1
        assert ws.target_size == (320, 240)


class TestCameraServiceLifecycle:
    """Tests for opening, availability validation, releasing, and context manager."""

    def test_open_with_injected_capture_device(self) -> None:
        mock_dev = MockCaptureSuccess(width=640, height=480)
        camera = CameraService(capture_device=mock_dev)
        assert camera.is_opened is True

        # Validation should succeed
        assert camera.validate_availability() is True

    def test_open_fails_when_camera_not_found(self) -> None:
        with patch("cv2.VideoCapture", return_value=MockCaptureFailOpen()):
            camera = CameraService(camera_index=99)
            with pytest.raises(CameraNotFoundError, match="Cannot open webcam at index 99"):
                camera.open()
            assert camera.is_opened is False

    def test_open_fails_when_camera_yields_empty_frame(self) -> None:
        with patch("cv2.VideoCapture", return_value=MockCaptureYieldsEmpty()):
            camera = CameraService(camera_index=0)
            with pytest.raises(CameraError, match="failed to yield initial test frame"):
                camera.open()
            assert camera.is_opened is False

    def test_context_manager_protocol(self) -> None:
        mock_dev = MockCaptureSuccess()
        with patch("cv2.VideoCapture", return_value=mock_dev):
            with CameraService(camera_index=0) as cam:
                assert cam.is_opened is True
                success, frame = cam.read_frame()
                assert success is True
                assert frame is not None

            # After exiting context, camera must be released
            assert cam.is_opened is False
            assert mock_dev.released is True

    def test_release_is_idempotent(self) -> None:
        mock_dev = MockCaptureSuccess()
        camera = CameraService(capture_device=mock_dev)
        assert camera.is_opened is True

        camera.release()
        assert camera.is_opened is False

        # Repeated calls should not raise
        camera.release()
        camera.close()
        assert camera.is_opened is False


class TestCameraServiceCapture:
    """Tests for frame capture, auto-resizing, and structured frame representations."""

    def test_read_frame_auto_resizes_to_target_resolution(self) -> None:
        # Device yields 1280x720, target is 640x480
        mock_dev = MockCaptureSuccess(width=1280, height=720)
        camera = CameraService(
            width=640,
            height=480,
            auto_resize=True,
            capture_device=mock_dev,
        )

        success, frame = camera.read_frame()
        assert success is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert camera.frame_count == 1

    def test_read_frame_without_resize_preserves_native_dimensions(self) -> None:
        mock_dev = MockCaptureSuccess(width=1280, height=720)
        camera = CameraService(
            width=640,
            height=480,
            auto_resize=False,
            capture_device=mock_dev,
        )

        success, frame = camera.read_frame()
        assert success is True
        assert frame is not None
        assert frame.shape == (720, 1280, 3)

    def test_read_convenience_method(self) -> None:
        mock_dev = MockCaptureSuccess(width=640, height=480)
        camera = CameraService(capture_device=mock_dev)

        frame = camera.read()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (config.frame_height, config.frame_width, 3)

    def test_read_structured_returns_camera_frame(self) -> None:
        mock_dev = MockCaptureSuccess(width=800, height=600)
        camera = CameraService(
            width=640,
            height=480,
            auto_resize=True,
            capture_device=mock_dev,
        )

        structured = camera.read_structured()
        assert structured is not None
        assert isinstance(structured, CameraFrame)
        assert structured.width == 640
        assert structured.height == 480
        assert structured.frame_number == 1
        assert structured.frame.shape == (480, 640, 3)

    def test_read_fails_when_device_closed(self) -> None:
        camera = CameraService()
        # Not opened yet
        success, frame = camera.read_frame()
        assert success is False
        assert frame is None

        structured = camera.read_structured()
        assert structured is None

        direct = camera.read()
        assert direct is None

    def test_read_fails_when_device_returns_false(self) -> None:
        mock_dev = MockCaptureYieldsEmpty()
        camera = CameraService(capture_device=mock_dev)

        success, frame = camera.read_frame()
        assert success is False
        assert frame is None

    def test_frame_count_increments_only_on_success(self) -> None:
        mock_dev = MockCaptureSuccess(width=640, height=480)
        camera = CameraService(capture_device=mock_dev)

        for expected in range(1, 6):
            cam_frame = camera.read_structured()
            assert cam_frame is not None
            assert cam_frame.frame_number == expected
            assert camera.frame_count == expected


class TestCameraAvailabilityProbes:
    """Tests for availability probing methods."""

    def test_is_camera_available_true_when_valid(self) -> None:
        mock_dev = MockCaptureSuccess(width=640, height=480)
        with patch("cv2.VideoCapture", return_value=mock_dev):
            assert CameraService.is_camera_available(0) is True

    def test_is_camera_available_false_when_unopened(self) -> None:
        mock_dev = MockCaptureFailOpen()
        with patch("cv2.VideoCapture", return_value=mock_dev):
            assert CameraService.is_camera_available(9) is False

    def test_is_camera_available_false_when_empty_frame(self) -> None:
        mock_dev = MockCaptureYieldsEmpty()
        with patch("cv2.VideoCapture", return_value=mock_dev):
            assert CameraService.is_camera_available(0) is False


class TestIndependenceFromRecognition:
    """Verify that CameraService has zero dependence on AI recognition services."""

    def test_camera_service_module_imports_no_ai(self) -> None:
        import app.vision.camera as cam_mod

        # Verify no torch, transformers, or clip references in camera module namespace
        assert not hasattr(cam_mod, "torch")
        assert not hasattr(cam_mod, "CLIPVisionModel")
        assert not hasattr(cam_mod, "LocalFaceEngine")
        assert not hasattr(cam_mod, "ObjectRecognitionService")
        assert not hasattr(cam_mod, "PersonRecognitionService")
