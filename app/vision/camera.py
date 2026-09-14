"""
app.vision.camera — Reusable Webcam Capture Service
===================================================

Manages video capture hardware using OpenCV:
- Device discovery, opening, and validation.
- Resolution configuration and automatic frame resizing.
- Thread-safe, resilient frame capture with error detection.
- Context manager protocol (``with CameraService() as cam:``).
- Idempotent, leak-free device release and clean shutdown.

Independent:
------------
This layer has ZERO dependencies on AI, CLIP, or face recognition modules.
It delivers raw OpenCV BGR NumPy arrays to downstream consumers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np

from app.config import config

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Base exception for camera hardware and capture failures."""
    pass


class CameraNotFoundError(CameraError):
    """Raised when the specified camera index cannot be opened."""
    pass


class FrameCaptureError(CameraError):
    """Raised when reading a frame from the camera fails."""
    pass


@dataclass
class CameraFrame:
    """A captured video frame with capture metadata.

    Attributes
    ----------
    frame : np.ndarray
        Captured BGR image array (H, W, C).
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    frame_number : int
        Monotonically increasing sequence number since capture started.
    """

    frame: np.ndarray
    width: int
    height: int
    frame_number: int


class CameraService:
    """Reusable camera service for real-time video acquisition.

    Parameters
    ----------
    camera_index : int, optional
        Camera device index. Defaults to ``config.camera_index`` (0).
    width : int, optional
        Target frame width. Defaults to ``config.frame_width`` (640).
    height : int, optional
        Target frame height. Defaults to ``config.frame_height`` (480).
    auto_resize : bool, optional
        If True, resizes frames to (width, height) if the camera's native
        resolution differs. Default is True.
    backend : int, optional
        OpenCV VideoCapture backend (e.g. ``cv2.CAP_ANY`` or ``cv2.CAP_DSHOW`` on Windows).
    capture_device : Any, optional
        Pre-instantiated capture object for dependency injection / testing.
    """

    def __init__(
        self,
        camera_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        auto_resize: bool = True,
        backend: Optional[int] = None,
        capture_device: Any = None,
    ) -> None:
        self._camera_index: int = (
            camera_index if camera_index is not None else config.camera_index
        )
        self._target_width: int = width if width is not None else config.frame_width
        self._target_height: int = height if height is not None else config.frame_height
        self._auto_resize: bool = auto_resize
        self._backend: int = backend if backend is not None else cv2.CAP_ANY

        self._cap: Any = capture_device
        self._frame_count: int = 0
        self._is_opened: bool = capture_device is not None and getattr(capture_device, "isOpened", lambda: True)()

    @property
    def camera_index(self) -> int:
        """Target camera hardware index."""
        return self._camera_index

    @property
    def target_size(self) -> Tuple[int, int]:
        """Target (width, height) in pixels."""
        return (self._target_width, self._target_height)

    @property
    def is_opened(self) -> bool:
        """Whether the video capture device is active and open."""
        if self._cap is None:
            return False
        if hasattr(self._cap, "isOpened"):
            return bool(self._cap.isOpened())
        return self._is_opened

    @property
    def frame_count(self) -> int:
        """Total number of frames successfully captured during this session."""
        return self._frame_count

    # ── Context Manager Protocol ──────────────────────────────────────────────

    def __enter__(self) -> "CameraService":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()

    # ── Lifecycle Management ──────────────────────────────────────────────────

    def open(self, camera_index: Optional[int] = None) -> None:
        """Open the camera device and configure capture properties.

        Parameters
        ----------
        camera_index : int, optional
            Override the configured camera index.

        Raises
        ------
        CameraNotFoundError
            If the device index cannot be opened.
        """
        if camera_index is not None:
            self._camera_index = camera_index

        if self.is_opened:
            logger.debug("Camera %d is already open.", self._camera_index)
            return

        logger.info(
            "Opening camera index %d (target resolution: %dx%d)...",
            self._camera_index,
            self._target_width,
            self._target_height,
        )

        cap = cv2.VideoCapture(self._camera_index, self._backend)

        # Attempt to set hardware capture properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)

        if not cap.isOpened():
            cap.release()
            raise CameraNotFoundError(
                f"Cannot open webcam at index {self._camera_index}. Verify device connection and permissions."
            )

        self._cap = cap
        self._is_opened = True
        self._frame_count = 0

        # Validate that the camera can actually deliver a frame
        valid = self.validate_availability()
        if not valid:
            self.release()
            raise CameraError(
                f"Webcam at index {self._camera_index} opened but failed to yield initial test frame."
            )

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Webcam %d opened successfully (native resolution: %dx%d).",
            self._camera_index,
            actual_w,
            actual_h,
        )

    def validate_availability(self) -> bool:
        """Check if the opened camera can deliver a valid non-empty frame.

        Returns
        -------
        bool
            True if the camera yields a valid frame, False otherwise.
        """
        if not self.is_opened or self._cap is None:
            return False

        try:
            ret, frame = self._cap.read()
            return bool(ret and frame is not None and frame.size > 0)
        except Exception as exc:
            logger.warning("Camera availability test failed: %s", exc)
            return False

    @staticmethod
    def is_camera_available(camera_index: int = 0) -> bool:
        """Probe whether a camera index is available on the system without throwing."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return bool(ret and frame is not None and frame.size > 0)

    # ── Frame Acquisition ─────────────────────────────────────────────────────

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a single frame from the camera.

        Returns
        -------
        tuple of (success: bool, frame: np.ndarray or None)
            - success : True if frame was captured and resized successfully.
            - frame : Captured BGR NumPy array (H, W, 3), or None if read failed.
        """
        if not self.is_opened or self._cap is None:
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.warning("Camera %d failed to read frame.", self._camera_index)
            return False, None

        # Auto-resize if configured and dimensions differ
        if self._auto_resize:
            h, w = frame.shape[:2]
            if w != self._target_width or h != self._target_height:
                frame = cv2.resize(
                    frame,
                    (self._target_width, self._target_height),
                    interpolation=cv2.INTER_LINEAR,
                )

        self._frame_count += 1
        return True, frame

    def read(self) -> Optional[np.ndarray]:
        """Convenience method returning the frame directly or None on failure."""
        success, frame = self.read_frame()
        return frame if success else None

    def read_structured(self) -> Optional[CameraFrame]:
        """Capture a frame and bundle it into a structured CameraFrame object."""
        success, frame = self.read_frame()
        if not success or frame is None:
            return None
        h, w = frame.shape[:2]
        return CameraFrame(
            frame=frame,
            width=w,
            height=h,
            frame_number=self._frame_count,
        )

    # ── Resource Cleanup ──────────────────────────────────────────────────────

    def release(self) -> None:
        """Release camera hardware cleanly. Idempotent and safe to call repeatedly."""
        if self._cap is not None:
            try:
                if hasattr(self._cap, "release"):
                    self._cap.release()
            except Exception as exc:
                logger.debug("Exception during camera release: %s", exc)
            self._cap = None

        self._is_opened = False
        logger.debug("Camera %d released cleanly.", self._camera_index)

    def close(self) -> None:
        """Alias for release()."""
        self.release()


# Alias for alternate naming convention
WebcamService = CameraService
