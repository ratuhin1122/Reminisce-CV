"""
app.speech.cooldown — Per-Memory Speech Cooldown & Announcement Controller
==========================================================================

Governs speech frequency to avoid overwhelming or repeating narrations:
- Enforces per-entity cooldowns (default: 30 seconds).
- Tracks speech event timestamps per memory.
- Allows immediate announcement for newly recognized entities.
- Permits re-announcement only after the cooldown period has elapsed.
- Prevents duplicate speech events while an utterance is in-progress/playing.
- Independent from the underlying TTS audio driver.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class SpeechEvent:
    """Historical record of an entity announcement.

    Attributes
    ----------
    entity_id : int
        Database ID of the announced entity.
    entity_type : str
        Type of entity (``"object"`` or ``"person"``).
    announced_at : float
        UNIX timestamp when announcement occurred.
    text : str
        Spoken utterance text.
    """

    entity_id: int
    entity_type: str
    announced_at: float
    text: str = ""


class SpeechCooldownController:
    """Controls announcement eligibility and prevents repetitive speech narration.

    Parameters
    ----------
    default_cooldown : float, optional
        Minimum duration in seconds before the same entity may be re-announced.
        Defaults to ``config.speech_cooldown_sec`` (30.0s).
    """

    def __init__(self, default_cooldown: Optional[float] = None) -> None:
        self.default_cooldown: float = (
            default_cooldown
            if default_cooldown is not None
            else config.speech_cooldown_sec
        )

        # Map of (entity_type, entity_id) -> last announced timestamp
        self._last_announced: Dict[Tuple[str, int], float] = {}

        # Set of (entity_type, entity_id) currently queued or speaking
        self._in_progress: Set[Tuple[str, int]] = set()

        # Audit history of speech events
        self._history: List[SpeechEvent] = []

    def _make_key(self, entity_id: int, entity_type: str = "object") -> Tuple[str, int]:
        return (entity_type.lower().strip(), entity_id)

    # ── Cooldown Queries ──────────────────────────────────────────────────────

    def can_announce(
        self,
        entity_id: int,
        entity_type: str = "object",
        current_time: Optional[float] = None,
        cooldown_override: Optional[float] = None,
    ) -> bool:
        """Check if an entity is eligible to be spoken.

        Parameters
        ----------
        entity_id : int
            Database memory ID.
        entity_type : str, optional
            ``"object"`` or ``"person"``. Default is ``"object"``.
        current_time : float, optional
            UNIX timestamp to test. Defaults to ``time.time()``.
        cooldown_override : float, optional
            Override the default cooldown duration for this check.

        Returns
        -------
        bool
            True if entity can be announced, False if on cooldown or currently speaking.
        """
        key = self._make_key(entity_id, entity_type)

        # 1. Prevent duplicate speech while speech for this entity is active/in-progress
        if key in self._in_progress:
            logger.debug("Announcement blocked: %s %d is already in progress.", entity_type, entity_id)
            return False

        # 2. Check cooldown timestamp
        last_time = self._last_announced.get(key)
        if last_time is None:
            # Newly observed entity — eligible immediately
            return True

        now = current_time if current_time is not None else time.time()
        eff_cooldown = (
            cooldown_override if cooldown_override is not None else self.default_cooldown
        )
        elapsed = now - last_time

        if elapsed >= eff_cooldown:
            return True

        logger.debug(
            "Announcement suppressed: %s %d is in cooldown (%.1fs remaining of %.1fs).",
            entity_type,
            entity_id,
            eff_cooldown - elapsed,
            eff_cooldown,
        )
        return False

    def get_remaining_cooldown(
        self,
        entity_id: int,
        entity_type: str = "object",
        current_time: Optional[float] = None,
        cooldown_override: Optional[float] = None,
    ) -> float:
        """Calculate remaining cooldown in seconds for an entity.

        Returns
        -------
        float
            Seconds remaining before entity can be announced, or 0.0 if ready.
        """
        key = self._make_key(entity_id, entity_type)
        last_time = self._last_announced.get(key)
        if last_time is None:
            return 0.0

        now = current_time if current_time is not None else time.time()
        eff_cooldown = (
            cooldown_override if cooldown_override is not None else self.default_cooldown
        )
        remaining = eff_cooldown - (now - last_time)
        return max(0.0, round(remaining, 2))

    def get_last_announced(
        self, entity_id: int, entity_type: str = "object"
    ) -> Optional[float]:
        """Return the timestamp of the last announcement for an entity, or None."""
        key = self._make_key(entity_id, entity_type)
        return self._last_announced.get(key)

    # ── State Updates ─────────────────────────────────────────────────────────

    def record_announcement(
        self,
        entity_id: int,
        entity_type: str = "object",
        current_time: Optional[float] = None,
        text: str = "",
    ) -> None:
        """Record that an announcement was spoken for an entity.

        Parameters
        ----------
        entity_id : int
            Database memory ID.
        entity_type : str, optional
            ``"object"`` or ``"person"``. Default is ``"object"``.
        current_time : float, optional
            Timestamp to record. Defaults to ``time.time()``.
        text : str, optional
            The spoken text.
        """
        now = current_time if current_time is not None else time.time()
        key = self._make_key(entity_id, entity_type)

        self._last_announced[key] = now
        self._history.append(
            SpeechEvent(
                entity_id=entity_id,
                entity_type=entity_type,
                announced_at=now,
                text=text,
            )
        )
        logger.debug("Recorded announcement for %s %d at timestamp %.3f", entity_type, entity_id, now)

    def should_announce(
        self,
        entity_id: int,
        entity_type: str = "object",
        current_time: Optional[float] = None,
        cooldown_override: Optional[float] = None,
        record: bool = True,
        text: str = "",
    ) -> bool:
        """Evaluate eligibility and optionally record announcement in one step.

        Parameters
        ----------
        entity_id : int
            Database memory ID.
        entity_type : str, optional
            ``"object"`` or ``"person"``. Default is ``"object"``.
        current_time : float, optional
            Observation timestamp.
        cooldown_override : float, optional
            Cooldown override.
        record : bool, optional
            If True and eligible, automatically records the announcement. Default True.
        text : str, optional
            Spoken utterance text if recorded.

        Returns
        -------
        bool
            True if announced/eligible, False if suppressed.
        """
        now = current_time if current_time is not None else time.time()
        if self.can_announce(
            entity_id=entity_id,
            entity_type=entity_type,
            current_time=now,
            cooldown_override=cooldown_override,
        ):
            if record:
                self.record_announcement(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    current_time=now,
                    text=text,
                )
            return True
        return False

    def mark_in_progress(self, entity_id: int, entity_type: str = "object") -> None:
        """Mark that speech synthesis is currently active for this entity."""
        key = self._make_key(entity_id, entity_type)
        self._in_progress.add(key)

    def mark_completed(self, entity_id: int, entity_type: str = "object") -> None:
        """Mark that speech synthesis has completed for this entity."""
        key = self._make_key(entity_id, entity_type)
        self._in_progress.discard(key)

    def reset(
        self, entity_id: Optional[int] = None, entity_type: Optional[str] = None
    ) -> None:
        """Reset cooldown tracking.

        Parameters
        ----------
        entity_id : int, optional
            If specified, resets cooldown for this entity ID only.
        entity_type : str, optional
            Required if entity_id is specified.
        """
        if entity_id is not None:
            t = entity_type or "object"
            key = self._make_key(entity_id, t)
            self._last_announced.pop(key, None)
            self._in_progress.discard(key)
        else:
            self._last_announced.clear()
            self._in_progress.clear()

    @property
    def history(self) -> List[SpeechEvent]:
        """Chronological audit trail of all recorded speech events."""
        return list(self._history)
