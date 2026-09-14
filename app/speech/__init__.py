"""
app.speech — Non-Blocking Text-to-Speech Output
=================================================

Responsibilities:
    - Offline TTS using pyttsx3
    - Background thread with speech queue
    - Never blocks the webcam processing loop
    - Graceful startup and shutdown
"""

from app.speech.cooldown import (
    SpeechCooldownController,
    SpeechEvent,
)
from app.speech.service import (
    MockSpeechEngine,
    SpeechService,
    SpeechTask,
)

__all__ = [
    "MockSpeechEngine",
    "SpeechCooldownController",
    "SpeechEvent",
    "SpeechService",
    "SpeechTask",
]


