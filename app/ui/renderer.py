"""
app.ui.renderer — OpenCV-Based Visual Overlay Renderer
========================================================

Renders structured visual overlays with clean aesthetic hierarchy on live video frames:
1. Top HUD bar:
   - Application branding ("ReminisceCV Assistant")
   - Recognition state badge (IDLE, ACQUIRING / CANDIDATE, STABLE)
   - Speech status indicator (MUTED, SPEAKING, COOLDOWN, READY)
   - Performance telemetry (real-time FPS, face & object latencies)
2. Bounding boxes & entity labels:
   - Familiar face bounding boxes (green for confirmed, amber for unconfirmed)
   - Object proposal bounding boxes with similarity percentage tags
3. Memory details card (lower HUD):
   - High-contrast semi-transparent acrylic dark card
   - Memory title / recognized entity name
   - Giver / person and occasion / year metadata
   - Stored personal narrative excerpt
   - Similarity / confidence score badge
4. Footer legend:
   - Keyboard control hints: [Q] Quit  [R] Reset  [M] Mute/Unmute
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from app.memory.retrieval import StructuredMemoryResponse
from app.recognition.pipeline import FrameRecognitionResult
from app.recognition.tracker import TemporalStatus

logger = logging.getLogger(__name__)


# ── Color Palette (BGR) ───────────────────────────────────────────────────────
COLOR_BG_DARK = (18, 18, 22)
COLOR_CARD_BORDER = (65, 65, 75)
COLOR_TEXT_PRIMARY = (250, 250, 250)
COLOR_TEXT_SECONDARY = (190, 195, 205)
COLOR_TEXT_MUTED = (130, 135, 145)

# Status Badge Colors
COLOR_GREEN = (65, 215, 95)       # Stable / Matched / Ready
COLOR_AMBER = (35, 175, 255)      # Candidate / Acquiring / Unknown
COLOR_BLUE = (220, 170, 75)       # Idle / Scanning
COLOR_RED = (60, 75, 235)         # Muted / Error
COLOR_CYAN = (245, 215, 45)       # Speaking


@dataclass
class SpeechUIState:
    """State information for speech audio status rendering."""

    is_muted: bool = False
    is_speaking: bool = False
    cooldown_remaining: float = 0.0
    last_spoken_text: str = ""


class UIRenderer:
    """Renders structured, high-contrast visual hierarchy on OpenCV BGR frames.

    Parameters
    ----------
    font : int
        OpenCV font face. Defaults to ``cv2.FONT_HERSHEY_SIMPLEX``.
    show_fps : bool
        Whether to render performance telemetry. Default True.
    """

    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        show_fps: bool = True,
    ) -> None:
        self.font = font
        self.show_fps = show_fps

    # ── Master Render Entry Point ─────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        result: Optional[FrameRecognitionResult] = None,
        speech_state: Optional[SpeechUIState] = None,
    ) -> np.ndarray:
        """Render complete UI overlay on top of the given video frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame.
        result : FrameRecognitionResult, optional
            Recognition output including faces, objects, tracking, and memory.
        speech_state : SpeechUIState, optional
            Current audio / speech status.

        Returns
        -------
        np.ndarray
            Annotated BGR frame with visual overlays applied.
        """
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()
        fh, fw = canvas.shape[:2]

        # 1. Draw bounding boxes for detected faces and objects
        if result is not None:
            self._draw_face_boxes(canvas, result)
            self._draw_object_boxes(canvas, result)

        # 2. Draw Top HUD Header (Brand, Tracking State, Speech Status, FPS)
        self._draw_header_hud(canvas, result, speech_state)

        # 3. Draw Memory Details Card (Bottom Overlay)
        self._draw_memory_card(canvas, result)

        # 4. Draw Footer Controls Legend
        self._draw_footer_legend(canvas)

        return canvas

    # ── Visual Components ─────────────────────────────────────────────────────

    def _draw_header_hud(
        self,
        canvas: np.ndarray,
        result: Optional[FrameRecognitionResult],
        speech_state: Optional[SpeechUIState],
    ) -> None:
        """Render the top HUD bar with state badges, speech status, and FPS."""
        fh, fw = canvas.shape[:2]
        hud_height = 54

        # Semi-transparent backdrop bar
        self._draw_alpha_rect(canvas, 0, 0, fw, hud_height, COLOR_BG_DARK, alpha=0.78)
        cv2.line(canvas, (0, hud_height), (fw, hud_height), COLOR_CARD_BORDER, 1)

        # App branding title
        cv2.putText(
            canvas,
            "ReminisceCV",
            (16, 34),
            self.font,
            0.65,
            COLOR_TEXT_PRIMARY,
            2,
            cv2.LINE_AA,
        )

        # Current recognition state determination
        state_text, state_color = self._get_recognition_state(result)
        self._draw_badge(
            canvas,
            text=state_text,
            x=160,
            y=14,
            bg_color=state_color,
            text_color=(15, 15, 20),
            padding=(10, 5),
            font_scale=0.45,
        )

        # Speech status indicator
        speech_x = 350
        if speech_state is not None:
            if speech_state.is_muted:
                speech_text = "SPEECH: MUTED"
                speech_color = COLOR_RED
            elif speech_state.is_speaking:
                speech_text = "SPEAKING..."
                speech_color = COLOR_CYAN
            elif speech_state.cooldown_remaining > 0.0:
                speech_text = f"COOLDOWN ({speech_state.cooldown_remaining:.1f}s)"
                speech_color = COLOR_AMBER
            else:
                speech_text = "SPEECH: READY"
                speech_color = COLOR_GREEN

            self._draw_badge(
                canvas,
                text=speech_text,
                x=speech_x,
                y=14,
                bg_color=speech_color,
                text_color=(15, 15, 20),
                padding=(8, 5),
                font_scale=0.42,
            )

        # Performance Telemetry (FPS & Latency) on the right
        if self.show_fps and result is not None:
            fps_text = f"FPS: {result.timing.fps:.1f}"
            lat_text = f"Lat: {result.timing.total_latency_ms:.0f}ms"
            telem_text = f"{fps_text} | {lat_text}"
            (tw, _), _ = cv2.getTextSize(telem_text, self.font, 0.45, 1)
            cv2.putText(
                canvas,
                telem_text,
                (fw - tw - 16, 33),
                self.font,
                0.45,
                COLOR_TEXT_SECONDARY,
                1,
                cv2.LINE_AA,
            )

    def _draw_face_boxes(
        self,
        canvas: np.ndarray,
        result: FrameRecognitionResult,
    ) -> None:
        """Render bounding boxes for all detected faces."""
        for person in result.detected_people:
            if not person.bbox:
                continue

            x, y, w, h = person.bbox.x, person.bbox.y, person.bbox.width, person.bbox.height
            color = COLOR_GREEN if person.matched else COLOR_AMBER

            # Draw sleek corner accents + box
            self._draw_corner_box(canvas, x, y, w, h, color, thickness=2, corner_len=14)

            # Label badge
            score_pct = int(person.similarity * 100) if person.similarity > 0 else 0
            label = f"{person.name} ({score_pct}%)"
            badge_y = max(18, y - 8)
            self._draw_badge(
                canvas,
                text=label,
                x=x,
                y=badge_y - 18,
                bg_color=color,
                text_color=(15, 15, 20),
                padding=(6, 3),
                font_scale=0.42,
            )

    def _draw_object_boxes(
        self,
        canvas: np.ndarray,
        result: FrameRecognitionResult,
    ) -> None:
        """Render bounding box for the top recognized object candidate."""
        top_obj = result.top_object
        if not top_obj or not top_obj.matched:
            return

        # Find matching candidate region if available
        matched_box: Optional[Tuple[int, int, int, int]] = None
        for cand in result.candidate_regions:
            # Prefer non-full-frame region when available
            if cand.region_type != "full_frame":
                matched_box = cand.bbox
                break

        if matched_box is not None:
            x, y, w, h = matched_box
            color = COLOR_GREEN
            self._draw_corner_box(canvas, x, y, w, h, color, thickness=2, corner_len=18)
            score_pct = int(top_obj.similarity * 100)
            label = f"{top_obj.name} ({score_pct}%)"
            badge_y = max(18, y - 8)
            self._draw_badge(
                canvas,
                text=label,
                x=x,
                y=badge_y - 18,
                bg_color=color,
                text_color=(15, 15, 20),
                padding=(6, 3),
                font_scale=0.42,
            )

    def _draw_memory_card(
        self,
        canvas: np.ndarray,
        result: Optional[FrameRecognitionResult],
    ) -> None:
        """Render the structured memory information card at the bottom of the screen."""
        if result is None:
            return

        fh, fw = canvas.shape[:2]
        margin = 16
        card_h = 108
        footer_h = 24
        card_y = fh - card_h - footer_h - 10
        card_w = fw - (2 * margin)

        # Retrieve memory details: prefer stable_memory, then recognized object/person memory
        mem: Optional[StructuredMemoryResponse] = result.stable_memory
        entity_name: str = ""
        similarity: float = 0.0
        is_stable: bool = False

        if result.tracker_state:
            is_stable = result.tracker_state.has_stable_match

        if mem and mem.found:
            entity_name = mem.title
            if result.tracker_state and result.tracker_state.object_state.candidate:
                similarity = result.tracker_state.object_state.candidate.similarity
            elif result.tracker_state and result.tracker_state.person_state.candidate:
                similarity = result.tracker_state.person_state.candidate.similarity
        else:
            # Check if active candidate or top match exists
            top_obj = result.top_object
            top_person = result.top_person
            if top_obj and top_obj.matched:
                entity_name = top_obj.name
                similarity = top_obj.similarity
                if top_obj.memory:
                    mem = StructuredMemoryResponse(
                        found=True,
                        entity_id=top_obj.memory.id,
                        entity_type="object",
                        title=top_obj.memory.title or top_obj.name,
                        giver=top_obj.memory.giver_name,
                        occasion=top_obj.memory.occasion,
                        year=top_obj.memory.year_or_date,
                        narrative=top_obj.memory.narrative_memory,
                        is_active=top_obj.memory.is_active,
                    )
            elif top_person and top_person.matched:
                entity_name = top_person.name
                similarity = top_person.similarity
                if top_person.memory:
                    mem = StructuredMemoryResponse(
                        found=True,
                        entity_id=top_person.memory.id,
                        entity_type="person",
                        title=top_person.name,
                        giver=top_person.relationship,
                        occasion=top_person.memory.occasion,
                        year=top_person.memory.year_or_date,
                        narrative=top_person.memory.narrative_memory,
                        is_active=top_person.memory.is_active,
                    )

        # If no recognized entity, do not clutter screen
        if not entity_name:
            return

        # Semi-transparent card background
        self._draw_alpha_rect(canvas, margin, card_y, card_w, card_h, COLOR_BG_DARK, alpha=0.85)
        # Card border
        border_color = COLOR_GREEN if is_stable else COLOR_AMBER
        cv2.rectangle(
            canvas,
            (margin, card_y),
            (margin + card_w, card_y + card_h),
            border_color,
            1,
        )

        # Title / Entity Name
        score_pct = int(similarity * 100) if similarity > 0 else 0
        title_str = f"MEMORY: {entity_name.upper()}"
        cv2.putText(
            canvas,
            title_str,
            (margin + 16, card_y + 26),
            self.font,
            0.58,
            COLOR_TEXT_PRIMARY,
            2,
            cv2.LINE_AA,
        )

        # Confidence tag
        score_badge = f"{score_pct}% Match"
        (bw, _), _ = cv2.getTextSize(score_badge, self.font, 0.45, 1)
        self._draw_badge(
            canvas,
            text=score_badge,
            x=margin + card_w - bw - 26,
            y=card_y + 10,
            bg_color=border_color,
            text_color=(15, 15, 20),
            padding=(6, 3),
            font_scale=0.42,
        )

        # Metadata Line (Giver / Person | Occasion | Year)
        meta_parts = []
        if mem:
            if mem.giver:
                label = "From" if mem.entity_type == "object" else "Relation"
                meta_parts.append(f"{label}: {mem.giver}")
            if mem.occasion:
                meta_parts.append(f"Occasion: {mem.occasion}")
            if mem.year:
                meta_parts.append(f"Year: {mem.year}")

        meta_line = "  |  ".join(meta_parts) if meta_parts else "Recognized personal entity"
        cv2.putText(
            canvas,
            meta_line,
            (margin + 16, card_y + 52),
            self.font,
            0.45,
            COLOR_AMBER if not is_stable else COLOR_TEXT_SECONDARY,
            1,
            cv2.LINE_AA,
        )

        # Narrative Story (Clean truncation)
        story_text = ""
        if mem and mem.narrative:
            story_text = f'"{mem.narrative}"'
        else:
            story_text = "Acquiring stability..." if not is_stable else "Personal memory verified."

        # Truncate narrative to fit width comfortably
        max_chars = max(40, int(card_w / 8.5))
        if len(story_text) > max_chars:
            story_text = story_text[: max_chars - 3] + "..."

        cv2.putText(
            canvas,
            story_text,
            (margin + 16, card_y + 78),
            self.font,
            0.44,
            COLOR_TEXT_SECONDARY if is_stable else COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        # Status note
        status_note = "Ready for narration" if is_stable else "Hold still to confirm recognition"
        cv2.putText(
            canvas,
            f"State: {status_note}",
            (margin + 16, card_y + 98),
            self.font,
            0.38,
            COLOR_GREEN if is_stable else COLOR_AMBER,
            1,
            cv2.LINE_AA,
        )

    def _draw_footer_legend(self, canvas: np.ndarray) -> None:
        """Render minimal keyboard controls hint at the bottom."""
        fh, fw = canvas.shape[:2]
        footer_h = 24
        footer_y = fh - footer_h

        # Translucent bar
        self._draw_alpha_rect(canvas, 0, footer_y, fw, footer_h, (12, 12, 15), alpha=0.88)

        legend_text = "Controls:  [Q] Quit    [R] Reset Recognition    [M] Mute / Unmute"
        (tw, _), _ = cv2.getTextSize(legend_text, self.font, 0.40, 1)
        start_x = max(16, (fw - tw) // 2)

        cv2.putText(
            canvas,
            legend_text,
            (start_x, footer_y + 16),
            self.font,
            0.40,
            COLOR_TEXT_SECONDARY,
            1,
            cv2.LINE_AA,
        )

    # ── Drawing Primitives ────────────────────────────────────────────────────

    def _get_recognition_state(
        self, result: Optional[FrameRecognitionResult]
    ) -> Tuple[str, Tuple[int, int, int]]:
        """Determine tracking status badge text and color."""
        if result is None or result.tracker_state is None:
            return "IDLE / SCANNING", COLOR_BLUE

        state = result.tracker_state
        if state.has_stable_match:
            # Retrieve stable entity label
            name = ""
            if state.object_state.is_stable and state.object_state.candidate:
                name = state.object_state.candidate.name
            elif state.person_state.is_stable and state.person_state.candidate:
                name = state.person_state.candidate.name
            return f"STABLE: {name.upper()}" if name else "STABLE", COLOR_GREEN

        has_candidate = (
            state.object_state.status == TemporalStatus.CANDIDATE
            or state.person_state.status == TemporalStatus.CANDIDATE
        )
        if has_candidate:
            cand = state.object_state.candidate or state.person_state.candidate
            name = cand.name if cand else "Entity"
            # Show stability progress
            elapsed = 0.0
            if state.object_state.status == TemporalStatus.CANDIDATE:
                elapsed = state.object_state.elapsed_duration_sec
            elif state.person_state.status == TemporalStatus.CANDIDATE:
                elapsed = state.person_state.elapsed_duration_sec
            return f"ACQUIRING: {name} ({elapsed:.1f}s)", COLOR_AMBER

        return "SCANNING", COLOR_BLUE

    def _draw_badge(
        self,
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (10, 10, 15),
        padding: Tuple[int, int] = (8, 4),
        font_scale: float = 0.42,
    ) -> None:
        """Render a rounded-corner filled badge with text."""
        pad_x, pad_y = padding
        (tw, th), baseline = cv2.getTextSize(text, self.font, font_scale, 1)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(canvas.shape[1] - 1, x1 + tw + 2 * pad_x)
        y2 = min(canvas.shape[0] - 1, y1 + th + 2 * pad_y)

        # Draw filled pill background
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_color, -1)

        # Draw badge label text
        text_x = x1 + pad_x
        text_y = y1 + pad_y + th
        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            self.font,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_corner_box(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int],
        thickness: int = 2,
        corner_len: int = 15,
    ) -> None:
        """Draw an elegant bounding box with pronounced corners."""
        fh, fw = canvas.shape[:2]
        x1 = max(0, min(x, fw - 1))
        y1 = max(0, min(y, fh - 1))
        x2 = max(0, min(x + w, fw - 1))
        y2 = max(0, min(y + h, fh - 1))

        # Thin base rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

        # Pronounced corner brackets
        clen = min(corner_len, (x2 - x1) // 3, (y2 - y1) // 3)
        if clen <= 0:
            return

        # Top-Left
        cv2.line(canvas, (x1, y1), (x1 + clen, y1), color, thickness)
        cv2.line(canvas, (x1, y1), (x1, y1 + clen), color, thickness)
        # Top-Right
        cv2.line(canvas, (x2, y1), (x2 - clen, y1), color, thickness)
        cv2.line(canvas, (x2, y1), (x2, y1 + clen), color, thickness)
        # Bottom-Left
        cv2.line(canvas, (x1, y2), (x1 + clen, y2), color, thickness)
        cv2.line(canvas, (x1, y2), (x1, y2 - clen), color, thickness)
        # Bottom-Right
        cv2.line(canvas, (x2, y2), (x2 - clen, y2), color, thickness)
        cv2.line(canvas, (x2, y2), (x2, y2 - clen), color, thickness)

    def _draw_alpha_rect(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int],
        alpha: float = 0.7,
    ) -> None:
        """Blends a semi-transparent filled rectangle onto the canvas."""
        fh, fw = canvas.shape[:2]
        x1 = max(0, min(x, fw - 1))
        y1 = max(0, min(y, fh - 1))
        x2 = max(0, min(x + w, fw - 1))
        y2 = max(0, min(y + h, fh - 1))

        if x2 <= x1 or y2 <= y1:
            return

        roi = canvas[y1:y2, x1:x2]
        overlay = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
        canvas[y1:y2, x1:x2] = roi
