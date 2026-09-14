"""
ReminisceCV — A Real-Time Vision-Based Personal Memory Retrieval
and Assistive System.

This package contains the core application modules organized by
responsibility:

    app.config        — Central configuration constants
    app.database      — SQLite persistent memory storage
    app.vision        — Pretrained vision model embeddings (CLIP)
    app.recognition   — Real-time recognition engine with debouncing
    app.memory        — Memory registration and retrieval logic
    app.speech        — Non-blocking text-to-speech output
    app.ui            — OpenCV-based on-screen display
    app.evaluation    — AI recognition evaluation framework
"""

__version__ = "0.1.0"
__project__ = "ReminisceCV"
