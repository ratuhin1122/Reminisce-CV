"""
app.recognition.pipeline — Real-Time Webcam Recognition Pipeline
================================================================

Connects real-time video capture with vision and face recognition engines:
1. Captures frames from ``CameraService`` (or accepts external frames).
2. Prepares and validates frame buffers.
3. Extracts candidate visual regions (full frame, center focus, foreground contours).
4. Runs local familiar face detection and recognition.
5. Runs embedding-based personal object recognition across candidate visual regions.
6. Gathers comprehensive timing metrics (capture time, stage latencies, FPS).
7. Produces structured ``FrameRecognitionResult`` representations.

Performance & Memory Guarantees:
--------------------------------
- Models are pre-loaded once during pipeline initialization.
- ZERO expensive model loading occurs inside the frame loop.
- The pipeline is fully modular: face and object stages can be enabled/disabled independently.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from app.config import config
from app.memory.retrieval import (
    MemoryRetrievalService,
    StructuredMemoryResponse,
)
from app.recognition.person_service import (
    PersonRecognitionResult,
    PersonRecognitionService,
)
from app.recognition.region_proposal import (
    CandidateRegion,
    CandidateRegionExtractor,
)
from app.recognition.service import (
    ObjectRecognitionService,
    RecognitionResult,
)
from app.recognition.tracker import (
    DualRecognitionState,
    DualRecognitionTracker,
    TrackedEntity,
)
from app.vision.camera import CameraFrame, CameraService

logger = logging.getLogger(__name__)


@dataclass
class PipelineTiming:
    """Microsecond-accurate latency and throughput metrics for a frame.

    Attributes
    ----------
    capture_time_ms : float
        Time taken by camera capture hardware.
    prep_time_ms : float
        Time spent validating and preparing the frame buffer and candidate regions.
    face_latency_ms : float
        Latency of face detection and embedding comparison.
    object_latency_ms : float
        Latency of object candidate embedding extraction and matching.
    total_latency_ms : float
        Total end-to-end processing latency for the frame.
    fps : float
        Instantaneous or running average frames-per-second throughput.
    """

    capture_time_ms: float = 0.0
    prep_time_ms: float = 0.0
    face_latency_ms: float = 0.0
    object_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    fps: float = 0.0


@dataclass
class FrameRecognitionResult:
    """Structured result of executing the recognition pipeline on a single frame.

    Attributes
    ----------
    frame_number : int
        Sequential frame identifier.
    timestamp : float
        UNIX timestamp when frame processing began.
    frame : np.ndarray
        Processed BGR frame array.
    detected_people : list of PersonRecognitionResult
        Face recognition outcomes for all faces detected in the frame.
    recognized_objects : list of RecognitionResult
        Object recognition outcomes for candidate visual regions.
    candidate_regions : list of CandidateRegion
        Visual regions of interest analyzed during this frame.
    timing : PipelineTiming
        Latency and throughput metrics.
    has_matches : bool
        True if at least one familiar person or personal object was matched.
    """

    frame_number: int
    timestamp: float
    frame: np.ndarray
    detected_people: List[PersonRecognitionResult] = field(default_factory=list)
    recognized_objects: List[RecognitionResult] = field(default_factory=list)
    candidate_regions: List[CandidateRegion] = field(default_factory=list)
    timing: PipelineTiming = field(default_factory=PipelineTiming)
    has_matches: bool = False
    tracker_state: Optional[DualRecognitionState] = None
    stable_memory: Optional[StructuredMemoryResponse] = None

    @property
    def is_stable(self) -> bool:
        """Whether any recognized entity has achieved temporal stability."""
        return self.tracker_state.has_stable_match if self.tracker_state else False

    @property
    def became_stable(self) -> bool:
        """Whether any recognized entity newly became stable on this frame."""
        return self.tracker_state.any_became_stable if self.tracker_state else False

    @property
    def top_person(self) -> Optional[PersonRecognitionResult]:
        """Return the best-matching recognized person in this frame, or None."""
        matched = [p for p in self.detected_people if p.matched]
        if not matched:
            return None
        return max(matched, key=lambda p: p.similarity)

    @property
    def top_object(self) -> Optional[RecognitionResult]:
        """Return the best-matching recognized object in this frame, or None."""
        matched = [o for o in self.recognized_objects if o.matched]
        if not matched:
            return None
        return max(matched, key=lambda o: o.similarity)


class RealTimeRecognitionPipeline:
    """Modular, real-time recognition pipeline connecting capture with AI engines.

    Parameters
    ----------
    camera_service : CameraService, optional
        Camera acquisition service. If None, instantiates default CameraService.
    object_service : ObjectRecognitionService, optional
        Personal object recognition service. If None, instantiates default service.
    person_service : PersonRecognitionService, optional
        Familiar person recognition service. If None, instantiates default service.
    region_extractor : CandidateRegionExtractor, optional
        Visual candidate region extractor. Defaults to CandidateRegionExtractor().
    enable_objects : bool, optional
        Whether to execute object recognition stage. Default True.
    enable_faces : bool, optional
        Whether to execute face recognition stage. Default True.
    auto_warmup : bool, optional
        Whether to pre-load all AI models and galleries during initialization. Default True.
    """

    def __init__(
        self,
        camera_service: Optional[CameraService] = None,
        object_service: Optional[ObjectRecognitionService] = None,
        person_service: Optional[PersonRecognitionService] = None,
        region_extractor: Optional[CandidateRegionExtractor] = None,
        tracker: Optional[DualRecognitionTracker] = None,
        memory_retrieval: Optional[MemoryRetrievalService] = None,
        enable_objects: bool = True,
        enable_faces: bool = True,
        auto_warmup: bool = True,
    ) -> None:
        self.camera: CameraService = camera_service or CameraService()
        self.object_service: ObjectRecognitionService = (
            object_service or ObjectRecognitionService()
        )
        self.person_service: PersonRecognitionService = (
            person_service or PersonRecognitionService()
        )
        self.region_extractor: CandidateRegionExtractor = (
            region_extractor or CandidateRegionExtractor()
        )
        self.tracker: DualRecognitionTracker = tracker or DualRecognitionTracker()
        self.memory_retrieval: MemoryRetrievalService = (
            memory_retrieval or MemoryRetrievalService()
        )

        self.enable_objects: bool = enable_objects
        self.enable_faces: bool = enable_faces

        self._frame_count: int = 0
        self._is_running: bool = False
        self._models_warmed_up: bool = False

        # FPS calculation tracking
        self._last_frame_timestamp: float = time.time()
        self._fps_smoothed: float = 0.0

        if auto_warmup:
            self.warmup()

    # ── Context Manager & Lifecycle ───────────────────────────────────────────

    def __enter__(self) -> "RealTimeRecognitionPipeline":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def warmup(self) -> None:
        """Pre-load all AI models and database galleries upfront.

        Ensures models are resident in memory and ready for zero-latency inference
        inside the real-time frame loop.
        """
        if self._models_warmed_up:
            return

        logger.info("Warming up ReminisceCV recognition models and galleries...")
        t0 = time.perf_counter()

        if self.enable_objects and self.object_service:
            if not self.object_service.vision_model.is_loaded:
                self.object_service.vision_model.load_model()
            self.object_service.refresh_gallery()

        if self.enable_faces and self.person_service:
            if not self.person_service.face_engine.is_loaded:
                self.person_service.face_engine.load_model()
            self.person_service.refresh_gallery()

        self._models_warmed_up = True
        logger.info(
            "Recognition pipeline warmed up successfully in %.2f seconds.",
            time.perf_counter() - t0,
        )

    def start(self) -> None:
        """Start the pipeline, opening the camera device and ensuring warmup."""
        self.warmup()
        if not self.camera.is_opened:
            self.camera.open()
        self._is_running = True
        self._last_frame_timestamp = time.time()
        logger.info("RealTimeRecognitionPipeline started.")

    def stop(self) -> None:
        """Stop capture loop and release camera hardware cleanly."""
        self._is_running = False
        if self.camera.is_opened:
            self.camera.release()
        logger.info("RealTimeRecognitionPipeline stopped and camera released.")

    def release(self) -> None:
        """Alias for stop()."""
        self.stop()

    # ── Frame Processing ──────────────────────────────────────────────────────

    def prepare_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Validate and standardize frame format for downstream inference.

        Parameters
        ----------
        raw_frame : np.ndarray
            Captured BGR array.

        Returns
        -------
        np.ndarray
            Standardized, contiguous BGR uint8 frame.
        """
        if raw_frame is None or raw_frame.size == 0:
            raise ValueError("Cannot prepare empty or None frame.")

        frame = np.ascontiguousarray(raw_frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        return frame

    def process_frame(
        self,
        frame: Optional[np.ndarray] = None,
        frame_number: Optional[int] = None,
    ) -> FrameRecognitionResult:
        """Execute one complete cycle of capture, preparation, and recognition.

        Parameters
        ----------
        frame : np.ndarray, optional
            If provided, uses this image directly rather than reading from camera.
            Useful for offline processing, unit testing, and benchmarking.
        frame_number : int, optional
            Override sequential frame count.

        Returns
        -------
        FrameRecognitionResult
            Structured result containing all detected faces, recognized objects,
            candidate regions, and timing telemetry.
        """
        t_start = time.perf_counter()
        t_capture_ms = 0.0

        # 1. Acquire Frame
        if frame is None:
            t_cap_0 = time.perf_counter()
            success, captured = self.camera.read_frame()
            t_capture_ms = (time.perf_counter() - t_cap_0) * 1000.0
            if not success or captured is None:
                raise RuntimeError("Failed to capture frame from camera device.")
            frame = captured

        self._frame_count += 1
        seq_num = frame_number if frame_number is not None else self._frame_count

        # 2. Prepare Frame & Extract Candidate Regions
        t_prep_0 = time.perf_counter()
        prepared_frame = self.prepare_frame(frame)
        candidate_regions: List[CandidateRegion] = []
        if self.enable_objects:
            candidate_regions = self.region_extractor.extract(prepared_frame)
        t_prep_ms = (time.perf_counter() - t_prep_0) * 1000.0

        # 3. Face Detection & Recognition
        detected_people: List[PersonRecognitionResult] = []
        t_face_ms = 0.0
        if self.enable_faces and self.person_service:
            t_face_0 = time.perf_counter()
            detected_people = self.person_service.detect_and_recognize_faces(
                prepared_frame
            )
            t_face_ms = (time.perf_counter() - t_face_0) * 1000.0

        # 4. Object Recognition across Candidate Regions
        recognized_objects: List[RecognitionResult] = []
        t_obj_ms = 0.0
        if self.enable_objects and self.object_service and candidate_regions:
            t_obj_0 = time.perf_counter()
            # Evaluate candidates; record top candidate or any match
            best_result: Optional[RecognitionResult] = None

            for region in candidate_regions:
                res = self.object_service.recognize(region.crop)
                if best_result is None or res.similarity > best_result.similarity:
                    best_result = res
                if res.matched:
                    recognized_objects.append(res)

            # If no candidate matched threshold, still include the best candidate result
            if not recognized_objects and best_result is not None:
                recognized_objects.append(best_result)

            t_obj_ms = (time.perf_counter() - t_obj_0) * 1000.0

        # 5. Timing & FPS Computation
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        now = time.time()
        delta_t = now - self._last_frame_timestamp
        self._last_frame_timestamp = now
        instant_fps = 1.0 / delta_t if delta_t > 0 else 0.0

        # Exponential moving average for smooth FPS
        if self._fps_smoothed == 0.0:
            self._fps_smoothed = instant_fps
        else:
            self._fps_smoothed = 0.85 * self._fps_smoothed + 0.15 * instant_fps

        timing = PipelineTiming(
            capture_time_ms=t_capture_ms,
            prep_time_ms=t_prep_ms,
            face_latency_ms=t_face_ms,
            object_latency_ms=t_obj_ms,
            total_latency_ms=t_total_ms,
            fps=round(self._fps_smoothed, 1),
        )

        # 5. Temporal Stability Tracking
        top_p = next((p for p in detected_people if p.matched), None)
        top_o = next((o for o in recognized_objects if o.matched), None)
        tracker_state = self.tracker.update(
            object_detection=top_o,
            person_detection=top_p,
            timestamp=now,
        )

        # 6. Stored Personal Memory Retrieval for Stable Entities
        stable_memory: Optional[StructuredMemoryResponse] = None
        if tracker_state and tracker_state.has_stable_match:
            stable_cand = None
            if tracker_state.object_state.is_stable and tracker_state.object_state.candidate:
                stable_cand = tracker_state.object_state.candidate
            elif tracker_state.person_state.is_stable and tracker_state.person_state.candidate:
                stable_cand = tracker_state.person_state.candidate

            if stable_cand and stable_cand.entity_id:
                stable_memory = self.memory_retrieval.retrieve_memory(stable_cand.entity_id)

        has_matches = any(p.matched for p in detected_people) or any(
            o.matched for o in recognized_objects
        )

        return FrameRecognitionResult(
            frame_number=seq_num,
            timestamp=now,
            frame=prepared_frame,
            detected_people=detected_people,
            recognized_objects=recognized_objects,
            candidate_regions=candidate_regions,
            timing=timing,
            has_matches=has_matches,
            tracker_state=tracker_state,
            stable_memory=stable_memory,
        )

    def stream(self) -> Generator[FrameRecognitionResult, None, None]:
        """Continuously capture and process frames in real time.

        Yields
        ------
        FrameRecognitionResult
            Structured result for each acquired frame.
        """
        if not self._is_running:
            self.start()

        while self._is_running:
            try:
                result = self.process_frame()
                yield result
            except Exception as exc:
                logger.error("Error during real-time pipeline execution: %s", exc)
                break
