"""
tests.test_recognition_tracker — Unit Tests for Temporal Recognition Stability
==============================================================================

Tests cover:
    - Stable recognition (entity recognized consistently >= stability_duration)
    - Unstable recognition (insufficient duration or fluctuating observations)
    - Changing from object A to object B (clean reset without false triggering)
    - Temporary detection loss (brief dropouts tolerated within grace period)
    - Consecutive observations and timing accumulation
    - Disappearance and clean reset when loss timeout is exceeded
    - Edge trigger became_stable fired only once upon achieving stability
    - DualRecognitionTracker handling both object and person channels
"""

import time
import pytest

from app.database.models import Memory
from app.recognition.service import RecognitionResult
from app.recognition.person_service import PersonRecognitionResult
from app.recognition.tracker import (
    DualRecognitionState,
    DualRecognitionTracker,
    RecognitionState,
    RecognitionTracker,
    TemporalStatus,
    TrackedEntity,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_object_result(
    entity_id: int,
    name: str = "Test Object",
    similarity: float = 0.85,
    matched: bool = True,
) -> RecognitionResult:
    return RecognitionResult(
        matched=matched,
        similarity=similarity,
        entity_id=entity_id if matched else None,
        entity_type="object",
        name=name if matched else None,
        memory=Memory(id=entity_id, name=name) if matched else None,
        threshold=0.25,
    )


def make_person_result(
    entity_id: int,
    name: str = "Test Person",
    similarity: float = 0.88,
    matched: bool = True,
) -> PersonRecognitionResult:
    return PersonRecognitionResult(
        matched=matched,
        name=name if matched else "unknown",
        similarity=similarity,
        entity_id=entity_id if matched else None,
        entity_type="person",
        threshold=0.40,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestTemporalRecognitionStability:
    """Unit tests for RecognitionTracker temporal stability."""

    def test_single_frame_is_not_stable(self) -> None:
        """System must NOT immediately trigger after a single successful frame."""
        tracker = RecognitionTracker(stability_duration=1.5, min_consecutive_frames=3)
        res = make_object_result(entity_id=1, name="Pocket Watch")

        t0 = 100.0
        state = tracker.update(res, timestamp=t0)

        assert state.status == TemporalStatus.CANDIDATE
        assert state.is_stable is False
        assert state.became_stable is False
        assert state.consecutive_frames == 1
        assert state.candidate is not None
        assert state.candidate.entity_id == 1
        assert state.candidate.name == "Pocket Watch"
        assert state.first_detected_at == t0
        assert state.last_detected_at == t0

    def test_stable_recognition_reached_after_duration(self) -> None:
        """Entity becomes stable once observed consistently >= stability_duration."""
        tracker = RecognitionTracker(stability_duration=1.5, min_consecutive_frames=3)
        res = make_object_result(entity_id=1, name="Pocket Watch")

        # Frame 1: t = 100.0 (Candidate)
        s1 = tracker.update(res, timestamp=100.0)
        assert s1.is_stable is False
        assert s1.became_stable is False

        # Frame 2: t = 100.8 (Candidate, 0.8s elapsed < 1.5s)
        s2 = tracker.update(res, timestamp=100.8)
        assert s2.is_stable is False
        assert s2.became_stable is False
        assert s2.consecutive_frames == 2

        # Frame 3: t = 101.4 (Candidate, 1.4s elapsed < 1.5s)
        s3 = tracker.update(res, timestamp=101.4)
        assert s3.is_stable is False
        assert s3.became_stable is False
        assert s3.consecutive_frames == 3

        # Frame 4: t = 101.6 (Stable! 1.6s elapsed >= 1.5s and >= 3 frames)
        s4 = tracker.update(res, timestamp=101.6)
        assert s4.status == TemporalStatus.STABLE
        assert s4.is_stable is True
        assert s4.became_stable is True  # Fired edge trigger
        assert s4.consecutive_frames == 4
        assert s4.elapsed_duration_sec >= 1.5

        # Frame 5: t = 102.0 (Remains stable, but became_stable is False)
        s5 = tracker.update(res, timestamp=102.0)
        assert s5.status == TemporalStatus.STABLE
        assert s5.is_stable is True
        assert s5.became_stable is False  # Does not re-trigger edge

    def test_unstable_recognition_insufficient_duration(self) -> None:
        """Fluctuating detection that disappears before 1.5s never becomes stable."""
        tracker = RecognitionTracker(
            stability_duration=1.5,
            min_consecutive_frames=3,
            max_missed_frames=2,
            loss_timeout_sec=0.5,
        )
        res = make_object_result(entity_id=1, name="Pocket Watch")

        # Two quick frames within 0.4s
        tracker.update(res, timestamp=100.0)
        s2 = tracker.update(res, timestamp=100.4)
        assert s2.is_stable is False

        # Detection stops for 3 frames (exceeding max_missed_frames)
        tracker.update(None, timestamp=100.6)
        tracker.update(None, timestamp=100.8)
        s_lost = tracker.update(None, timestamp=101.0)

        assert s_lost.status == TemporalStatus.LOST
        assert s_lost.is_stable is False
        assert tracker.candidate is None

    def test_changing_from_object_a_to_object_b(self) -> None:
        """Switching target from Object A to Object B resets temporal counter.

        Object B must accumulate its own 1.5s before becoming stable, preventing
        false triggers from transient noisy classifications.
        """
        tracker = RecognitionTracker(stability_duration=1.5, min_consecutive_frames=3)
        obj_a = make_object_result(entity_id=1, name="Vintage Watch")
        obj_b = make_object_result(entity_id=2, name="Silver Medal")

        # Object A is tracked for 1.2s (still candidate)
        tracker.update(obj_a, timestamp=100.0)
        tracker.update(obj_a, timestamp=100.6)
        s_a = tracker.update(obj_a, timestamp=101.2)
        assert s_a.candidate.name == "Vintage Watch"
        assert s_a.consecutive_frames == 3
        assert s_a.is_stable is False

        # Suddenly Object B is detected at t = 101.5
        s_b1 = tracker.update(obj_b, timestamp=101.5)
        # Tracker should switch candidate to Object B and reset frame count
        assert s_b1.candidate.name == "Silver Medal"
        assert s_b1.candidate.entity_id == 2
        assert s_b1.consecutive_frames == 1
        assert s_b1.first_detected_at == 101.5
        assert s_b1.is_stable is False
        assert s_b1.became_stable is False

        # Object B continues for 1.0s (total elapsed 1.0s < 1.5s)
        s_b2 = tracker.update(obj_b, timestamp=102.5)
        assert s_b2.is_stable is False

        # Object B reaches 1.6s elapsed since its first detection (101.5 -> 103.1)
        s_b3 = tracker.update(obj_b, timestamp=103.1)
        assert s_b3.status == TemporalStatus.STABLE
        assert s_b3.is_stable is True
        assert s_b3.became_stable is True
        assert s_b3.candidate.name == "Silver Medal"

    def test_temporary_detection_loss_grace_period(self) -> None:
        """A single missed frame (e.g. motion blur/lighting flicker) does not reset candidate."""
        tracker = RecognitionTracker(
            stability_duration=1.5,
            min_consecutive_frames=3,
            max_missed_frames=2,
            loss_timeout_sec=0.8,
        )
        res = make_object_result(entity_id=1, name="Pocket Watch")

        # Observed at t=100.0 and t=100.5
        tracker.update(res, timestamp=100.0)
        s2 = tracker.update(res, timestamp=100.5)
        assert s2.consecutive_frames == 2
        assert s2.missed_frames == 0

        # Frame 3 (t=100.7): Temporary dropout (e.g. camera glitch or crop miss)
        s_drop = tracker.update(None, timestamp=100.7)
        # Candidate is preserved within grace period!
        assert s_drop.candidate is not None
        assert s_drop.candidate.name == "Pocket Watch"
        assert s_drop.missed_frames == 1
        assert s_drop.status == TemporalStatus.CANDIDATE

        # Frame 4 (t=101.0): Object is detected again!
        s_recover = tracker.update(res, timestamp=101.0)
        assert s_recover.missed_frames == 0
        assert s_recover.consecutive_frames == 3
        # First detected time was preserved!
        assert s_recover.first_detected_at == 100.0

        # Frame 5 (t=101.6): Total elapsed time is 1.6s >= 1.5s -> reaches stable!
        s_stable = tracker.update(res, timestamp=101.6)
        assert s_stable.is_stable is True
        assert s_stable.became_stable is True

    def test_unmatched_result_treated_as_no_detection(self) -> None:
        """A result with matched=False is treated as absence of candidate."""
        tracker = RecognitionTracker(stability_duration=1.5)
        unmatched = make_object_result(entity_id=1, matched=False)

        state = tracker.update(unmatched, timestamp=100.0)
        assert state.status == TemporalStatus.IDLE
        assert state.candidate is None
        assert state.is_stable is False

    def test_reset_clears_all_state(self) -> None:
        tracker = RecognitionTracker(stability_duration=1.5, min_consecutive_frames=2)
        res = make_object_result(entity_id=1, name="Watch")
        tracker.update(res, timestamp=100.0)
        tracker.update(res, timestamp=102.0)

        assert tracker.is_stable is True
        tracker.reset()

        assert tracker.status == TemporalStatus.IDLE
        assert tracker.is_stable is False
        assert tracker.candidate is None
        assert tracker.consecutive_frames == 0


class TestDualRecognitionTracker:
    """Unit tests for concurrent object and person tracking channels."""

    def test_dual_tracking_concurrent_channels(self) -> None:
        dual = DualRecognitionTracker(
            object_stability_sec=1.5,
            person_stability_sec=1.5,
            min_consecutive_frames=3,
        )

        obj = make_object_result(entity_id=10, name="Vintage Book")
        person = make_person_result(entity_id=20, name="Granddaughter")

        # Frame 1: t = 100.0
        s1 = dual.update(object_detection=obj, person_detection=person, timestamp=100.0)
        assert s1.has_stable_match is False
        assert s1.any_became_stable is False

        # Frame 2: t = 101.0
        s2 = dual.update(object_detection=obj, person_detection=person, timestamp=101.0)
        assert s2.has_stable_match is False

        # Frame 3: t = 101.6 (both reach stability >= 1.5s)
        s3 = dual.update(object_detection=obj, person_detection=person, timestamp=101.6)
        assert s3.has_stable_match is True
        assert s3.any_became_stable is True
        assert s3.object_state.is_stable is True
        assert s3.person_state.is_stable is True
        assert s3.object_state.candidate.name == "Vintage Book"
        assert s3.person_state.candidate.name == "Granddaughter"
