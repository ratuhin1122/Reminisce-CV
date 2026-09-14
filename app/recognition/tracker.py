"""
app.recognition.tracker — Temporal Recognition Stability & Debouncing
=====================================================================

Prevents false triggers and ensures recognition stability over time:
- A recognized person or object is NOT triggered immediately after a single frame.
- Recognition becomes stable only when the same entity is recognized consistently
  with similarity above threshold for the configured duration (default: 1.5s).
- Buffers against transient detection dropouts (temporary detection loss).
- Handles candidate tracking, consecutive observations, state transitions,
  and clean resets upon departure or target change.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from app.config import config
from app.database.models import Memory
from app.recognition.person_service import PersonRecognitionResult
from app.recognition.service import RecognitionResult

logger = logging.getLogger(__name__)


class TemporalStatus(str, Enum):
    """Lifecycle state of an entity recognition track."""

    IDLE = "idle"
    CANDIDATE = "candidate"
    STABLE = "stable"
    LOST = "lost"


@dataclass
class TrackedEntity:
    """Canonical representation of an entity being temporally tracked.

    Attributes
    ----------
    entity_id : int
        Database memory ID of the object or person.
    entity_type : str
        Entity classification: ``"object"`` or ``"person"``.
    name : str
        Display name of the entity.
    similarity : float
        Latest similarity score in [-1.0, 1.0].
    memory : Memory or None
        Complete memory record if available.
    reference_image_path : str or None
        Path to reference image matching the entity.
    extra : dict
        Additional metadata (e.g. relationship, bounding box).
    """

    entity_id: int
    entity_type: str
    name: str
    similarity: float
    memory: Optional[Memory] = None
    reference_image_path: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls, result: Union[RecognitionResult, PersonRecognitionResult, "TrackedEntity"]
    ) -> Optional["TrackedEntity"]:
        """Convert a recognition result into a TrackedEntity if matched."""
        if result is None:
            return None

        if isinstance(result, TrackedEntity):
            return result

        if not getattr(result, "matched", False):
            return None

        entity_id = getattr(result, "entity_id", None)
        if entity_id is None:
            return None

        entity_type = getattr(result, "entity_type", "object")
        name = getattr(result, "name", "unknown") or "unknown"
        similarity = float(getattr(result, "similarity", 0.0))
        memory = getattr(result, "memory", None)
        ref_path = getattr(result, "reference_image_path", None)

        extra: dict[str, Any] = {}
        if hasattr(result, "relationship") and result.relationship:
            extra["relationship"] = result.relationship
        if hasattr(result, "bbox") and result.bbox:
            extra["bbox"] = result.bbox

        return cls(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            similarity=similarity,
            memory=memory,
            reference_image_path=ref_path,
            extra=extra,
        )


@dataclass
class RecognitionState:
    """Temporal tracker state snapshot for a single tracking channel.

    Attributes
    ----------
    status : TemporalStatus
        Current tracking status (IDLE, CANDIDATE, STABLE, or LOST).
    candidate : TrackedEntity or None
        The entity currently under evaluation or stable.
    first_detected_at : float or None
        UNIX timestamp of the first consecutive observation.
    last_detected_at : float or None
        UNIX timestamp of the most recent observation.
    consecutive_frames : int
        Number of consecutive frames the entity has been observed.
    missed_frames : int
        Number of consecutive frames the entity was undetected.
    elapsed_duration_sec : float
        Elapsed seconds since first observation.
    is_stable : bool
        True if the candidate has satisfied stability duration and frame count.
    became_stable : bool
        True ONLY on the exact frame transition when stability is first reached.
    became_lost : bool
        True on the exact frame when an entity transitions to LOST.
    """

    status: TemporalStatus = TemporalStatus.IDLE
    candidate: Optional[TrackedEntity] = None
    first_detected_at: Optional[float] = None
    last_detected_at: Optional[float] = None
    consecutive_frames: int = 0
    missed_frames: int = 0
    elapsed_duration_sec: float = 0.0
    is_stable: bool = False
    became_stable: bool = False
    became_lost: bool = False


class RecognitionTracker:
    """Tracks recognition consistency over time to ensure temporal stability.

    Guarantees:
    - Avoids false triggers from single noisy frames.
    - Requires consistent observations for ``stability_duration`` seconds.
    - Tolerates brief dropouts up to ``max_missed_frames`` without losing state.
    - Accurately reports ``became_stable`` edge trigger for downstream notification.

    Parameters
    ----------
    stability_duration : float, optional
        Required duration in seconds for an entity to become stable.
        Defaults to ``config.recognition_stability_sec`` (1.5s).
    min_consecutive_frames : int, optional
        Minimum number of consecutive frames required alongside time duration.
        Defaults to ``config.debounce_frames`` (3).
    max_missed_frames : int, optional
        Maximum consecutive missed frames permitted before abandoning candidate.
        Default is 2 frames.
    loss_timeout_sec : float, optional
        Maximum elapsed duration without detection before resetting. Default 0.8s.
    """

    def __init__(
        self,
        stability_duration: Optional[float] = None,
        min_consecutive_frames: Optional[int] = None,
        max_missed_frames: int = 2,
        loss_timeout_sec: float = 0.8,
    ) -> None:
        self.stability_duration: float = (
            stability_duration
            if stability_duration is not None
            else config.recognition_stability_sec
        )
        self.min_consecutive_frames: int = (
            min_consecutive_frames
            if min_consecutive_frames is not None
            else config.debounce_frames
        )
        self.max_missed_frames: int = max_missed_frames
        self.loss_timeout_sec: float = loss_timeout_sec

        # State attributes
        self._candidate: Optional[TrackedEntity] = None
        self._first_detected_at: Optional[float] = None
        self._last_detected_at: Optional[float] = None
        self._consecutive_frames: int = 0
        self._missed_frames: int = 0
        self._status: TemporalStatus = TemporalStatus.IDLE
        self._was_stable: bool = False

    @property
    def status(self) -> TemporalStatus:
        """Current temporal state."""
        return self._status

    @property
    def is_stable(self) -> bool:
        """Whether the current candidate is in a verified stable state."""
        return self._status == TemporalStatus.STABLE

    @property
    def candidate(self) -> Optional[TrackedEntity]:
        """Currently tracked candidate entity."""
        return self._candidate

    @property
    def consecutive_frames(self) -> int:
        """Consecutive frames observed for current candidate."""
        return self._consecutive_frames

    @property
    def missed_frames(self) -> int:
        """Consecutive frames missed since last detection."""
        return self._missed_frames

    def reset(self) -> None:
        """Reset tracker to IDLE state."""
        self._candidate = None
        self._first_detected_at = None
        self._last_detected_at = None
        self._consecutive_frames = 0
        self._missed_frames = 0
        self._status = TemporalStatus.IDLE
        self._was_stable = False

    def update(
        self,
        detection: Optional[Union[RecognitionResult, PersonRecognitionResult, TrackedEntity]],
        timestamp: Optional[float] = None,
    ) -> RecognitionState:
        """Update tracker state with a new frame's recognition result.

        Parameters
        ----------
        detection : RecognitionResult, PersonRecognitionResult, TrackedEntity, or None
            Result from object or face recognition for the current frame.
        timestamp : float, optional
            Observation timestamp in seconds. Defaults to ``time.time()``.

        Returns
        -------
        RecognitionState
            Updated state containing status, candidate, elapsed time, and edge flags.
        """
        now = timestamp if timestamp is not None else time.time()
        tracked = TrackedEntity.from_result(detection)

        became_stable = False
        became_lost = False

        # ── Case 1: No Matched Detection in this Frame ────────────────────────
        if tracked is None:
            if self._candidate is not None:
                self._missed_frames += 1
                time_since_last = now - (self._last_detected_at or now)

                # Check if tolerance threshold is exceeded
                if (
                    self._missed_frames > self.max_missed_frames
                    or time_since_last > self.loss_timeout_sec
                ):
                    became_lost = self._was_stable or self._status == TemporalStatus.STABLE
                    lost_candidate = self._candidate
                    self.reset()
                    self._status = TemporalStatus.LOST
                    return RecognitionState(
                        status=TemporalStatus.LOST,
                        candidate=lost_candidate,
                        first_detected_at=None,
                        last_detected_at=None,
                        consecutive_frames=0,
                        missed_frames=self._missed_frames,
                        elapsed_duration_sec=0.0,
                        is_stable=False,
                        became_stable=False,
                        became_lost=became_lost,
                    )
                else:
                    # Within grace period: maintain candidate and status
                    elapsed = now - (self._first_detected_at or now)
                    return RecognitionState(
                        status=self._status,
                        candidate=self._candidate,
                        first_detected_at=self._first_detected_at,
                        last_detected_at=self._last_detected_at,
                        consecutive_frames=self._consecutive_frames,
                        missed_frames=self._missed_frames,
                        elapsed_duration_sec=round(max(0.0, elapsed), 3),
                        is_stable=self._status == TemporalStatus.STABLE,
                        became_stable=False,
                        became_lost=False,
                    )
            else:
                self._status = TemporalStatus.IDLE
                return RecognitionState(status=TemporalStatus.IDLE)

        # ── Case 2: Matched Detection Present ─────────────────────────────────
        is_same_entity = (
            self._candidate is not None
            and self._candidate.entity_id == tracked.entity_id
            and self._candidate.entity_type == tracked.entity_type
        )

        if not is_same_entity:
            # Different entity detected (or first entity after IDLE)
            if self._candidate is not None and (
                self._status == TemporalStatus.STABLE or self._was_stable
            ):
                became_lost = True

            # Initialize new candidate
            self._candidate = tracked
            self._first_detected_at = now
            self._last_detected_at = now
            self._consecutive_frames = 1
            self._missed_frames = 0
            self._status = TemporalStatus.CANDIDATE
            self._was_stable = False
            elapsed = 0.0

        else:
            # Same entity consistently observed
            self._candidate = tracked  # update with latest similarity/crop
            self._last_detected_at = now
            self._consecutive_frames += 1
            self._missed_frames = 0

            elapsed = now - (self._first_detected_at or now)

            # Evaluate stability condition
            if (
                elapsed >= self.stability_duration
                and self._consecutive_frames >= self.min_consecutive_frames
            ):
                if self._status != TemporalStatus.STABLE:
                    self._status = TemporalStatus.STABLE
                    self._was_stable = True
                    became_stable = True
                    logger.debug(
                        "Temporal stability reached for %s '%s' (ID %d) after %.2fs (%d frames)",
                        tracked.entity_type,
                        tracked.name,
                        tracked.entity_id,
                        elapsed,
                        self._consecutive_frames,
                    )
            else:
                self._status = TemporalStatus.CANDIDATE

        return RecognitionState(
            status=self._status,
            candidate=self._candidate,
            first_detected_at=self._first_detected_at,
            last_detected_at=self._last_detected_at,
            consecutive_frames=self._consecutive_frames,
            missed_frames=self._missed_frames,
            elapsed_duration_sec=round(max(0.0, elapsed), 3),
            is_stable=self._status == TemporalStatus.STABLE,
            became_stable=became_stable,
            became_lost=became_lost,
        )


@dataclass
class DualRecognitionState:
    """Aggregated temporal states for both object and person tracking channels."""

    object_state: RecognitionState
    person_state: RecognitionState

    @property
    def has_stable_match(self) -> bool:
        """True if either the object or person channel has reached a stable state."""
        return self.object_state.is_stable or self.person_state.is_stable

    @property
    def any_became_stable(self) -> bool:
        """True on the exact frame when either channel newly becomes stable."""
        return self.object_state.became_stable or self.person_state.became_stable


class DualRecognitionTracker:
    """Manages simultaneous temporal stability tracking for both objects and people."""

    def __init__(
        self,
        object_stability_sec: Optional[float] = None,
        person_stability_sec: Optional[float] = None,
        min_consecutive_frames: Optional[int] = None,
        max_missed_frames: int = 2,
    ) -> None:
        self.object_tracker = RecognitionTracker(
            stability_duration=object_stability_sec,
            min_consecutive_frames=min_consecutive_frames,
            max_missed_frames=max_missed_frames,
        )
        self.person_tracker = RecognitionTracker(
            stability_duration=person_stability_sec,
            min_consecutive_frames=min_consecutive_frames,
            max_missed_frames=max_missed_frames,
        )

    def reset(self) -> None:
        """Reset both object and person tracking channels."""
        self.object_tracker.reset()
        self.person_tracker.reset()

    def update(
        self,
        object_detection: Optional[Union[RecognitionResult, TrackedEntity]],
        person_detection: Optional[Union[PersonRecognitionResult, TrackedEntity]],
        timestamp: Optional[float] = None,
    ) -> DualRecognitionState:
        """Update both trackers with current frame detections."""
        now = timestamp if timestamp is not None else time.time()
        obj_state = self.object_tracker.update(object_detection, timestamp=now)
        person_state = self.person_tracker.update(person_detection, timestamp=now)
        return DualRecognitionState(object_state=obj_state, person_state=person_state)
