"""
app.ui.app — ReminisceCV Real-Time Memory Assistant Application
===============================================================

Orchestrates real-time video capture, AI recognition, temporal tracking,
personal memory retrieval, speech narration, and OpenCV visual feedback:

1. Feeds live camera frames into ``RealTimeRecognitionPipeline``.
2. Overlays rich visual hierarchy using ``UIRenderer``.
3. On achieving temporal stability, triggers non-blocking speech narration
   via ``SpeechService`` respecting ``SpeechCooldownController`` rules.
4. Processes keyboard inputs in real-time without latency:
   - ``Q`` or ``ESC``: Cleanly shuts down and releases all devices.
   - ``R``: Manually resets recognition tracking state.
   - ``M``: Toggles speech audio narration (Mute / Unmute).
5. Headless and mock support for deterministic testing and automation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from app.config import config
from app.memory.retrieval import StructuredMemoryResponse
from app.recognition.pipeline import FrameRecognitionResult, RealTimeRecognitionPipeline
from app.speech.cooldown import SpeechCooldownController
from app.speech.service import SpeechService
from app.ui.renderer import SpeechUIState, UIRenderer

logger = logging.getLogger(__name__)


class MemoryAssistantApp:
    """Primary OpenCV application controller for ReminisceCV.

    Parameters
    ----------
    pipeline : RealTimeRecognitionPipeline, optional
        Underlying video capture and AI recognition pipeline.
    speech_service : SpeechService, optional
        Asynchronous Text-to-Speech service.
    cooldown_controller : SpeechCooldownController, optional
        Per-memory cooldown governor.
    renderer : UIRenderer, optional
        OpenCV visual overlay engine.
    window_name : str, optional
        OpenCV display window title.
    start_muted : bool, optional
        Whether speech is initially muted. Default False.
    """

    def __init__(
        self,
        pipeline: Optional[RealTimeRecognitionPipeline] = None,
        speech_service: Optional[SpeechService] = None,
        cooldown_controller: Optional[SpeechCooldownController] = None,
        renderer: Optional[UIRenderer] = None,
        window_name: str = "ReminisceCV - Real-Time Memory Assistant",
        start_muted: bool = False,
    ) -> None:
        self.pipeline: RealTimeRecognitionPipeline = (
            pipeline or RealTimeRecognitionPipeline()
        )
        self.speech_service: SpeechService = speech_service or SpeechService()
        self.cooldown: SpeechCooldownController = (
            cooldown_controller or SpeechCooldownController()
        )
        self.renderer: UIRenderer = renderer or UIRenderer()

        self.window_name: str = window_name
        self.is_muted: bool = start_muted
        self.is_running: bool = False
        self._window_initialized: bool = False

        self.last_spoken_text: str = ""
        self.last_announced_entity_id: Optional[int] = None
        self.frame_count: int = 0

    # ── Context Manager & Lifecycle ───────────────────────────────────────────

    def __enter__(self) -> "MemoryAssistantApp":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        """Initialize components and start video capture."""
        if self.is_running:
            return

        logger.info("Starting ReminisceCV Memory Assistant UI application...")
        self.pipeline.start()
        self.speech_service.start()
        self.is_running = True

    def stop(self) -> None:
        """Cleanly release camera, speech worker, and OpenCV display windows."""
        if not self.is_running and not self._window_initialized:
            return

        self.is_running = False
        logger.info("Stopping ReminisceCV Memory Assistant UI application...")

        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as exc:
                logger.debug("Error stopping pipeline: %s", exc)

        if self.speech_service:
            try:
                self.speech_service.shutdown()
            except Exception as exc:
                logger.debug("Error stopping speech service: %s", exc)

        if self._window_initialized:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self._window_initialized = False

        logger.info("ReminisceCV Memory Assistant shutdown complete.")

    # ── Interactive Control Actions ───────────────────────────────────────────

    def reset_recognition(self) -> None:
        """Reset the temporal recognition tracker (Triggered by 'R')."""
        logger.info("Manual recognition reset requested (Key: 'R'). Clearing tracker state.")
        if self.pipeline and self.pipeline.tracker:
            self.pipeline.tracker.reset()

    def toggle_mute(self) -> bool:
        """Toggle audio speech narration on/off (Triggered by 'M').

        Returns
        -------
        bool
            New mute state (True if muted, False if unmuted).
        """
        self.is_muted = not self.is_muted
        state_str = "MUTED" if self.is_muted else "UNMUTED"
        logger.info("Speech narration %s (Key: 'M').", state_str)

        # If muted while speech is playing, clear pending queue
        if self.is_muted and self.speech_service:
            self.speech_service.stop()

        return self.is_muted

    def handle_key(self, key: int) -> bool:
        """Process keyboard input.

        Parameters
        ----------
        key : int
            ASCII / OpenCV key code.

        Returns
        -------
        bool
            True if application should continue running, False if quit was requested.
        """
        if key in (ord("q"), ord("Q"), 27):  # 'Q', 'q', or ESC
            logger.info("Quit requested via keyboard (Key: 'Q' / ESC).")
            self.is_running = False
            return False

        if key in (ord("r"), ord("R")):  # 'R' or 'r'
            self.reset_recognition()
            return True

        if key in (ord("m"), ord("M")):  # 'M' or 'm'
            self.toggle_mute()
            return True

        return True

    # ── Single Frame Processing Cycle ─────────────────────────────────────────

    def process_step(
        self,
        frame: Optional[np.ndarray] = None,
        headless: bool = False,
    ) -> Tuple[FrameRecognitionResult, np.ndarray]:
        """Execute one complete frame cycle: recognition, speech, overlay, display.

        Parameters
        ----------
        frame : np.ndarray, optional
            Direct input frame (useful for testing or video file replay).
        headless : bool, optional
            If True, skips ``cv2.imshow`` and ``cv2.waitKey``.

        Returns
        -------
        Tuple[FrameRecognitionResult, np.ndarray]
            (recognition_result, annotated_display_frame)
        """
        # 1. Run capture, recognition, tracking, and memory retrieval
        result = self.pipeline.process_frame(frame=frame)
        self.frame_count += 1

        # 2. Check for newly stable entities to trigger speech narration
        self._evaluate_speech_trigger(result)

        # 3. Compute speech UI state for overlay rendering
        speech_state = self._build_speech_ui_state(result)

        # 4. Render visual hierarchy overlay onto frame
        annotated_frame = self.renderer.render(
            frame=result.frame,
            result=result,
            speech_state=speech_state,
        )

        # 5. Handle OpenCV window presentation and keyboard input
        if not headless:
            self._display_and_poll(annotated_frame)

        return result, annotated_frame

    # ── Internal Speech & Display Helpers ─────────────────────────────────────

    def _evaluate_speech_trigger(self, result: FrameRecognitionResult) -> None:
        """Trigger non-blocking speech announcement when an entity becomes stable."""
        if not result.became_stable or not result.tracker_state:
            return

        # Determine the newly stable entity
        state = result.tracker_state
        entity_id: Optional[int] = None
        entity_type: str = "object"
        entity_name: str = ""

        if state.object_state.is_stable and state.object_state.candidate:
            cand = state.object_state.candidate
            entity_id = cand.entity_id
            entity_type = "object"
            entity_name = cand.name
        elif state.person_state.is_stable and state.person_state.candidate:
            cand = state.person_state.candidate
            entity_id = cand.entity_id
            entity_type = "person"
            entity_name = cand.name

        if entity_id is None:
            return

        # Check if muted
        if self.is_muted:
            logger.debug(
                "Speech narration suppressed: Muted (Entity: %s %d).",
                entity_type,
                entity_id,
            )
            return

        # Check cooldown governor
        if not self.cooldown.can_announce(entity_id=entity_id, entity_type=entity_type):
            logger.debug(
                "Speech narration suppressed: In cooldown (Entity: %s %d).",
                entity_type,
                entity_id,
            )
            return

        # Prepare narration text
        speech_text = self._build_narration_text(result, entity_name, entity_type)
        if not speech_text:
            return

        # Record announcement in cooldown tracker
        self.cooldown.record_announcement(
            entity_id=entity_id,
            entity_type=entity_type,
            text=speech_text,
        )

        # Queue speech asynchronously (worker handles audio non-blockingly)
        logger.info(
            "Announcing stable memory for %s %d: '%s'",
            entity_type,
            entity_id,
            speech_text,
        )
        self.speech_service.speak(speech_text)
        self.last_spoken_text = speech_text
        self.last_announced_entity_id = entity_id

    def _build_narration_text(
        self,
        result: FrameRecognitionResult,
        entity_name: str,
        entity_type: str,
    ) -> str:
        """Construct a natural, concise spoken utterance from personal memory."""
        mem = result.stable_memory
        if mem and mem.found:
            # If a stored personal narrative is available, prioritize it
            if mem.narrative and mem.narrative.strip():
                # Form clean prefix
                prefix = f"{mem.title}."
                if mem.giver:
                    giver_label = "given by" if entity_type == "object" else "related as"
                    prefix = f"{mem.title}, {giver_label} {mem.giver}."
                return f"{prefix} {mem.narrative}"

            # Fallback to metadata description
            parts = [mem.title]
            if mem.giver:
                parts.append(f"from {mem.giver}")
            if mem.occasion:
                parts.append(f"for {mem.occasion}")
            if mem.year:
                parts.append(f"in {mem.year}")
            return " ".join(parts) + "."

        # Generic confirmation if memory metadata is absent
        return f"Recognized {entity_name}."

    def _build_speech_ui_state(self, result: FrameRecognitionResult) -> SpeechUIState:
        """Compute the current speech status for the UI overlay."""
        remaining_cd = 0.0

        # Calculate remaining cooldown for active entity if present
        if result.tracker_state:
            active_id = None
            active_type = "object"
            if result.tracker_state.object_state.candidate:
                active_id = result.tracker_state.object_state.candidate.entity_id
                active_type = "object"
            elif result.tracker_state.person_state.candidate:
                active_id = result.tracker_state.person_state.candidate.entity_id
                active_type = "person"

            if active_id is not None:
                remaining_cd = self.cooldown.get_remaining_cooldown(
                    entity_id=active_id,
                    entity_type=active_type,
                )

        return SpeechUIState(
            is_muted=self.is_muted,
            is_speaking=self.speech_service.is_speaking,
            cooldown_remaining=remaining_cd,
            last_spoken_text=self.last_spoken_text,
        )

    def _display_and_poll(self, frame: np.ndarray) -> None:
        """Display frame in OpenCV window and poll keyboard events."""
        if not self._window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._window_initialized = True

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Key was pressed
            self.handle_key(key)

    # ── Main Application Loop ─────────────────────────────────────────────────

    def run(
        self,
        max_frames: Optional[int] = None,
        headless: bool = False,
    ) -> None:
        """Run the main OpenCV video capture and assistant loop.

        Parameters
        ----------
        max_frames : int, optional
            If set, exits cleanly after processing this number of frames.
        headless : bool, optional
            If True, runs without opening an OpenCV GUI window.
        """
        self.start()
        logger.info(
            "Entering ReminisceCV assistant main loop (headless=%s, max_frames=%s).",
            headless,
            max_frames,
        )

        try:
            while self.is_running:
                self.process_step(headless=headless)

                if max_frames and self.frame_count >= max_frames:
                    logger.info("Reached maximum requested frames (%d). Exiting.", max_frames)
                    break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting.")
        finally:
            self.stop()
