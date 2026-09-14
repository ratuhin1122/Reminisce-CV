"""
app.ui — OpenCV-Based User Interface & Memory Assistant HUD
===========================================================

Provides the real-time visual interface and application controller for ReminisceCV:
- ``UIRenderer``: High-contrast visual hierarchy overlays (HUD, bounding boxes, memory card).
- ``SpeechUIState``: Encapsulates speech status (muted, speaking, cooldown) for UI display.
- ``MemoryAssistantApp``: Main OpenCV application orchestrator with keyboard controls (Q, R, M).
"""

from app.ui.app import MemoryAssistantApp
from app.ui.renderer import (
    COLOR_AMBER,
    COLOR_BLUE,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    SpeechUIState,
    UIRenderer,
)

__all__ = [
    "MemoryAssistantApp",
    "UIRenderer",
    "SpeechUIState",
    "COLOR_GREEN",
    "COLOR_AMBER",
    "COLOR_BLUE",
    "COLOR_RED",
    "COLOR_CYAN",
]
