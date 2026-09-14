"""
app.speech.service — Non-Blocking Text-to-Speech Service
========================================================

Provides asynchronous speech synthesis using ``pyttsx3``:
- Dedicated background worker thread processing a FIFO speech queue.
- Never blocks the video capture or AI recognition loop.
- Serializes utterances to strictly prevent overlapping speech.
- Supports configurable speech rate and volume.
- Gracefully handles missing audio drivers or TTS initialization errors.
- Idempotent, clean shutdown and queue flushing mechanisms.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class SpeechTask:
    """An utterance request queued for speech synthesis.

    Attributes
    ----------
    text : str
        Text to be spoken.
    created_at : float
        UNIX timestamp when the task was created.
    """

    text: str
    created_at: float = field(default_factory=time.time)


class MockSpeechEngine:
    """Test double simulating pyttsx3.Engine for offline/headless verification."""

    def __init__(self, rate: int = 160, volume: float = 0.9) -> None:
        self.rate = rate
        self.volume = volume
        self.spoken_texts: List[str] = []
        self._is_stopped = False
        self._properties: dict[str, Any] = {
            "rate": rate,
            "volume": volume,
            "voice": "default",
        }

    def say(self, text: str) -> None:
        self.spoken_texts.append(text)

    def runAndWait(self) -> None:
        # Simulate brief synthesis latency in tests
        time.sleep(0.01)

    def stop(self) -> None:
        self._is_stopped = True

    def setProperty(self, name: str, value: Any) -> None:
        self._properties[name] = value
        if name == "rate":
            self.rate = int(value)
        elif name == "volume":
            self.volume = float(value)

    def getProperty(self, name: str) -> Any:
        return self._properties.get(name)


class SpeechService:
    """Asynchronous, non-blocking Text-to-Speech service using a dedicated worker thread.

    Parameters
    ----------
    rate : int, optional
        Speech rate in words per minute. Defaults to ``config.tts_rate`` (160).
    volume : float, optional
        Volume in [0.0, 1.0]. Defaults to ``config.tts_volume`` (0.9).
    voice_id : str, optional
        Optional specific voice identifier.
    auto_start : bool, optional
        Whether to start the background worker thread immediately. Default True.
    engine : Any, optional
        Pre-instantiated TTS engine (e.g. ``MockSpeechEngine`` for testing).
    """

    def __init__(
        self,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        voice_id: Optional[str] = None,
        auto_start: bool = True,
        engine: Any = None,
    ) -> None:
        self._target_rate: int = rate if rate is not None else config.tts_rate
        self._target_volume: float = (
            volume if volume is not None else config.tts_volume
        )
        self._voice_id: Optional[str] = voice_id

        self._engine: Any = engine
        self._is_available: bool = False
        self._is_speaking: bool = False

        self._queue: queue.Queue[Optional[SpeechTask]] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._init_lock = threading.Lock()

        # Initialize engine and start worker thread
        self._initialize_engine()

        if auto_start and self._is_available:
            self.start()

    # ── Engine Initialization ─────────────────────────────────────────────────

    def _initialize_engine(self) -> None:
        """Initialize the pyttsx3 engine safely, handling environment/audio failures."""
        if self._engine is not None:
            self._is_available = True
            return

        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self._target_rate)
            engine.setProperty("volume", self._target_volume)

            if self._voice_id:
                engine.setProperty("voice", self._voice_id)

            self._engine = engine
            self._is_available = True
            logger.info(
                "pyttsx3 TTS engine initialized successfully (rate=%d, volume=%.1f).",
                self._target_rate,
                self._target_volume,
            )
        except Exception as exc:
            self._is_available = False
            self._engine = None
            logger.warning(
                "Failed to initialize pyttsx3 TTS engine: %s. Speech output disabled.",
                exc,
            )

    # ── Worker Thread Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Start the background speech worker thread."""
        with self._init_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return

            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="ReminisceCV-SpeechWorker",
                daemon=True,
            )
            self._worker_thread.start()
            logger.debug("Speech worker thread started.")

    def _worker_loop(self) -> None:
        """Background thread consuming queued speech tasks sequentially."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Sentinel for shutdown
            if task is None:
                self._queue.task_done()
                break

            try:
                self._is_speaking = True
                self._execute_speech(task.text)
            except Exception as exc:
                logger.error("Error speaking text '%s': %s", task.text, exc)
            finally:
                self._is_speaking = False
                self._queue.task_done()

    def _execute_speech(self, text: str) -> None:
        """Synthesize speech on the worker thread."""
        if not self._engine or not text:
            return

        clean_text = text.strip()
        if not clean_text:
            return

        logger.debug("Synthesizing speech: '%s'", clean_text)
        self._engine.say(clean_text)
        self._engine.runAndWait()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Whether TTS engine is available and functional."""
        return self._is_available

    @property
    def is_speaking(self) -> bool:
        """Whether the engine is actively synthesizing speech."""
        return self._is_speaking

    @property
    def queue_size(self) -> int:
        """Number of speech tasks currently waiting in queue."""
        return self._queue.qsize()

    @property
    def rate(self) -> int:
        """Current speech rate in words per minute."""
        return self._target_rate

    @rate.setter
    def rate(self, value: int) -> None:
        """Set speech rate."""
        self._target_rate = max(50, min(value, 400))
        if self._engine and hasattr(self._engine, "setProperty"):
            try:
                self._engine.setProperty("rate", self._target_rate)
            except Exception as exc:
                logger.debug("Could not set rate on engine: %s", exc)

    @property
    def volume(self) -> float:
        """Current volume in [0.0, 1.0]."""
        return self._target_volume

    @volume.setter
    def volume(self, value: float) -> None:
        """Set volume."""
        self._target_volume = max(0.0, min(value, 1.0))
        if self._engine and hasattr(self._engine, "setProperty"):
            try:
                self._engine.setProperty("volume", self._target_volume)
            except Exception as exc:
                logger.debug("Could not set volume on engine: %s", exc)

    def speak(self, text: str, interrupt: bool = False) -> bool:
        """Queue text for non-blocking speech synthesis.

        Parameters
        ----------
        text : str
            The utterance text.
        interrupt : bool, optional
            If True, flushes pending queued speech and stops current speech
            before queuing this task. Default is False.

        Returns
        -------
        bool
            True if task was queued, False if text is empty or service is unavailable.
        """
        if not text or not text.strip():
            return False

        if not self._is_available:
            logger.debug("SpeechService unavailable. Skipping text: '%s'", text)
            return False

        if interrupt:
            self.stop()

        task = SpeechTask(text=text.strip())
        self._queue.put(task)
        return True

    def stop(self) -> None:
        """Clear all pending queued utterances and interrupt active speech."""
        # Empty queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break

        if self._engine and hasattr(self._engine, "stop"):
            try:
                self._engine.stop()
            except Exception as exc:
                logger.debug("Exception stopping engine: %s", exc)

    def shutdown(self, wait: bool = True, timeout: float = 2.0) -> None:
        """Cleanly stop worker thread and release TTS resources."""
        self.stop()
        self._stop_event.set()

        # Send poison pill sentinel
        self._queue.put(None)

        if wait and self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        logger.debug("SpeechService shut down cleanly.")

    # ── Context Manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "SpeechService":
        if not (self._worker_thread and self._worker_thread.is_alive()):
            self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    def __del__(self) -> None:
        self.shutdown(wait=False)
