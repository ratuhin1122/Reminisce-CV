"""
tests.test_speech_cooldown — Unit Tests for Speech Cooldown & Announcement Control
==================================================================================

Tests cover:
    - First announcement: newly recognized entity is immediately eligible
    - Repeated detection during cooldown: suppressed within configured duration
    - Announcement after cooldown: re-announcement permitted once duration expires
    - Switching between entities: cooldown for Entity A does not block Entity B
    - In-progress prevention: suppresses duplicate triggers while utterance is playing
    - Remaining cooldown calculations
    - Clean reset and history audit trail
"""

import time
import pytest

from app.config import config
from app.speech import (
    SpeechCooldownController,
    SpeechEvent,
)


@pytest.fixture
def cooldown_controller() -> SpeechCooldownController:
    """Controller with standard 30-second cooldown."""
    return SpeechCooldownController(default_cooldown=30.0)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSpeechCooldownControl:
    """Tests for per-entity cooldown enforcement."""

    def test_first_announcement_is_eligible_immediately(
        self, cooldown_controller: SpeechCooldownController
    ) -> None:
        """A new entity is immediately eligible for announcement on first detection."""
        t0 = 1000.0
        # Check eligibility
        assert cooldown_controller.can_announce(entity_id=1, entity_type="object", current_time=t0) is True
        assert cooldown_controller.get_remaining_cooldown(entity_id=1, entity_type="object", current_time=t0) == 0.0

        # Should announce and record
        announced = cooldown_controller.should_announce(
            entity_id=1,
            entity_type="object",
            current_time=t0,
            text="This is Grandfather's watch.",
        )
        assert announced is True
        assert cooldown_controller.get_last_announced(entity_id=1, entity_type="object") == t0
        assert len(cooldown_controller.history) == 1
        assert cooldown_controller.history[0].text == "This is Grandfather's watch."

    def test_repeated_detection_during_cooldown_is_suppressed(
        self, cooldown_controller: SpeechCooldownController
    ) -> None:
        """Same entity detected across subsequent frames within 30s is suppressed."""
        t0 = 1000.0
        # First announcement at t0
        cooldown_controller.record_announcement(entity_id=1, entity_type="object", current_time=t0)

        # Frame 2: t = 1000.033 (~30ms later)
        assert cooldown_controller.can_announce(entity_id=1, entity_type="object", current_time=1000.033) is False
        assert cooldown_controller.should_announce(entity_id=1, entity_type="object", current_time=1000.033) is False

        # Frame 100: t = 1015.0 (15s later < 30s)
        assert cooldown_controller.can_announce(entity_id=1, entity_type="object", current_time=1015.0) is False
        rem = cooldown_controller.get_remaining_cooldown(entity_id=1, entity_type="object", current_time=1015.0)
        assert rem == 15.0

        # History should still only have the initial event
        assert len(cooldown_controller.history) == 1

    def test_announcement_after_cooldown_expires(
        self, cooldown_controller: SpeechCooldownController
    ) -> None:
        """Entity can be re-announced once cooldown duration (30s) has passed."""
        t0 = 1000.0
        cooldown_controller.record_announcement(entity_id=1, entity_type="object", current_time=t0)

        # At t = 1029.9s (just before 30s) -> still suppressed
        assert cooldown_controller.can_announce(entity_id=1, entity_type="object", current_time=1029.9) is False

        # At t = 1030.0s (exactly 30s later) -> eligible!
        assert cooldown_controller.can_announce(entity_id=1, entity_type="object", current_time=1030.0) is True
        assert cooldown_controller.get_remaining_cooldown(entity_id=1, entity_type="object", current_time=1030.0) == 0.0

        # Re-announce
        re_announced = cooldown_controller.should_announce(
            entity_id=1,
            entity_type="object",
            current_time=1030.0,
            text="Second reminder: Grandfather's watch.",
        )
        assert re_announced is True
        assert len(cooldown_controller.history) == 2
        assert cooldown_controller.get_last_announced(entity_id=1, entity_type="object") == 1030.0

    def test_switching_between_entities(
        self, cooldown_controller: SpeechCooldownController
    ) -> None:
        """Entity A being on cooldown does NOT block Entity B from announcing."""
        t0 = 1000.0
        # Entity A announced at t0
        cooldown_controller.record_announcement(entity_id=10, entity_type="object", current_time=t0)
        assert cooldown_controller.can_announce(entity_id=10, entity_type="object", current_time=t0 + 5.0) is False

        # Entity B observed at t0 + 5.0s (Entity B has never been announced)
        assert cooldown_controller.can_announce(entity_id=20, entity_type="object", current_time=t0 + 5.0) is True

        # Announce Entity B
        announced_b = cooldown_controller.should_announce(
            entity_id=20,
            entity_type="object",
            current_time=t0 + 5.0,
            text="Silver Medal.",
        )
        assert announced_b is True

        # Both entities are now on independent cooldowns
        assert cooldown_controller.can_announce(entity_id=10, entity_type="object", current_time=t0 + 10.0) is False
        assert cooldown_controller.can_announce(entity_id=20, entity_type="object", current_time=t0 + 10.0) is False

        # Person vs Object with same ID are also tracked independently
        assert cooldown_controller.can_announce(entity_id=10, entity_type="person", current_time=t0 + 10.0) is True

    def test_prevent_duplicate_speech_when_in_progress(
        self, cooldown_controller: SpeechCooldownController
    ) -> None:
        """Entity currently queued or playing cannot trigger concurrent speech."""
        t0 = 1000.0
        # Mark entity 5 in progress
        cooldown_controller.mark_in_progress(entity_id=5, entity_type="person")

        # Even though never announced before, in-progress prevents duplicate
        assert cooldown_controller.can_announce(entity_id=5, entity_type="person", current_time=t0) is False

        # Once finished playing
        cooldown_controller.mark_completed(entity_id=5, entity_type="person")
        assert cooldown_controller.can_announce(entity_id=5, entity_type="person", current_time=t0) is True

    def test_reset_clears_cooldown(
        self, cooldown_controller: SpeechCooldownController
    ) -> None:
        """Reset clears cooldown for a specific entity or all entities."""
        t0 = 1000.0
        cooldown_controller.record_announcement(entity_id=1, entity_type="object", current_time=t0)
        cooldown_controller.record_announcement(entity_id=2, entity_type="object", current_time=t0)

        # Reset single entity
        cooldown_controller.reset(entity_id=1, entity_type="object")
        assert cooldown_controller.can_announce(entity_id=1, entity_type="object", current_time=t0 + 1.0) is True
        assert cooldown_controller.can_announce(entity_id=2, entity_type="object", current_time=t0 + 1.0) is False

        # Reset all
        cooldown_controller.reset()
        assert cooldown_controller.can_announce(entity_id=2, entity_type="object", current_time=t0 + 1.0) is True
