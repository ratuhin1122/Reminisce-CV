"""
tests.test_speech_service — Unit Tests for Local Text-to-Speech Service
========================================================================

Tests cover:
    - Default initialization from config (rate, volume)
    - Property setters for speech rate and volume
    - Non-blocking asynchronous speak() call
    - Rejection of empty or whitespace-only utterances
    - Sequential FIFO processing preventing overlapping speech
    - Interruption handling (clearing pending queue)
    - Stop and clean shutdown lifecycle (worker thread termination)
    - Context manager compliance
    - Graceful degradation on TTS initialization failure
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.config import config
from app.speech import (
    MockSpeechEngine,
    SpeechService,
    SpeechTask,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_engine() -> MockSpeechEngine:
    return MockSpeechEngine(rate=160, volume=0.9)


@pytest.fixture
def speech_service(mock_engine: MockSpeechEngine) -> SpeechService:
    service = SpeechService(engine=mock_engine, auto_start=True)
    yield service
    service.shutdown(wait=True)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSpeechServiceInitialization:
    """Tests verifying initialization defaults and engine configuration."""

    def test_default_rate_and_volume_from_config(self, mock_engine: MockSpeechEngine) -> None:
        service = SpeechService(engine=mock_engine)
        assert service.rate == config.tts_rate
        assert service.volume == config.tts_volume
        assert service.is_available is True
        service.shutdown()

    def test_custom_rate_and_volume_properties(self, mock_engine: MockSpeechEngine) -> None:
        service = SpeechService(rate=180, volume=0.75, engine=mock_engine)
        assert service.rate == 180
        assert service.volume == 0.75

        # Update properties dynamically
        service.rate = 200
        service.volume = 0.5
        assert service.rate == 200
        assert service.volume == 0.5
        assert mock_engine.rate == 200
        assert mock_engine.volume == 0.5
        service.shutdown()

    def test_rate_clamping(self, mock_engine: MockSpeechEngine) -> None:
        service = SpeechService(engine=mock_engine)
        service.rate = 10  # too low
        assert service.rate == 50

        service.rate = 900  # too high
        assert service.rate == 400
        service.shutdown()

    def test_volume_clamping(self, mock_engine: MockSpeechEngine) -> None:
        service = SpeechService(engine=mock_engine)
        service.volume = -0.5
        assert service.volume == 0.0

        service.volume = 2.0
        assert service.volume == 1.0
        service.shutdown()


class TestSpeechExecutionAndConcurrency:
    """Tests verifying non-blocking asynchronous speech and queuing behavior."""

    def test_speak_is_non_blocking(self, speech_service: SpeechService) -> None:
        t0 = time.perf_counter()
        queued = speech_service.speak("Hello world, this is a non-blocking test.")
        elapsed = time.perf_counter() - t0

        assert queued is True
        # speak() must return almost instantly (< 5ms)
        assert elapsed < 0.05

    def test_speak_rejects_empty_strings(self, speech_service: SpeechService) -> None:
        assert speech_service.speak("") is False
        assert speech_service.speak("   ") is False
        assert speech_service.speak(None) is False  # type: ignore[arg-type]

    def test_sequential_utterances_prevent_overlapping(
        self, speech_service: SpeechService, mock_engine: MockSpeechEngine
    ) -> None:
        """Queue guarantees FIFO execution without overlapping synthesis."""
        speech_service.speak("Utterance 1")
        speech_service.speak("Utterance 2")
        speech_service.speak("Utterance 3")

        # Wait for worker thread to process tasks
        for _ in range(50):
            if len(mock_engine.spoken_texts) >= 3:
                break
            time.sleep(0.02)

        assert mock_engine.spoken_texts == ["Utterance 1", "Utterance 2", "Utterance 3"]

    def test_interrupt_flushes_pending_queue(
        self, speech_service: SpeechService, mock_engine: MockSpeechEngine
    ) -> None:
        # Fill queue with low priority items
        speech_service.speak("Pending item 1")
        speech_service.speak("Pending item 2")
        speech_service.speak("Pending item 3")

        # Interrupt with urgent alert
        speech_service.speak("Urgent alert!", interrupt=True)

        # Wait for completion
        for _ in range(50):
            if "Urgent alert!" in mock_engine.spoken_texts:
                break
            time.sleep(0.02)

        assert "Urgent alert!" in mock_engine.spoken_texts
        assert mock_engine._is_stopped is True


class TestLifecycleAndShutdown:
    """Tests for clean shutdown, stop, and error handling."""

    def test_stop_clears_queue(self, speech_service: SpeechService) -> None:
        speech_service.speak("A")
        speech_service.speak("B")
        speech_service.stop()

        assert speech_service.queue_size == 0

    def test_shutdown_terminates_worker_thread(self, mock_engine: MockSpeechEngine) -> None:
        service = SpeechService(engine=mock_engine, auto_start=True)
        assert service._worker_thread is not None
        assert service._worker_thread.is_alive()

        service.shutdown(wait=True, timeout=1.0)
        assert not service._worker_thread.is_alive()

    def test_context_manager_protocol(self, mock_engine: MockSpeechEngine) -> None:
        with SpeechService(engine=mock_engine, auto_start=True) as service:
            assert service.is_available is True
            service.speak("Context test")

        assert not service._worker_thread.is_alive()

    def test_initialization_error_handled_gracefully(self) -> None:
        """When pyttsx3 fails to initialize, service degrades gracefully without raising."""
        with patch("pyttsx3.init", side_effect=RuntimeError("No audio device found")):
            service = SpeechService(engine=None, auto_start=True)
            assert service.is_available is False
            # speak() should return False safely
            assert service.speak("Test") is False
            service.shutdown()
