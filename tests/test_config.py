"""
tests.test_config — Verify configuration system
=================================================

Tests that the Config dataclass loads correctly with safe defaults,
environment variable overrides work, and the summary is renderable.
"""

import os
from pathlib import Path

import pytest

from app.config import Config, config, PROJECT_ROOT


class TestConfigDefaults:
    """Verify all default values are sensible."""

    def test_project_root_is_absolute(self) -> None:
        assert config.project_root.is_absolute()

    def test_data_dir_under_project(self) -> None:
        assert str(config.data_dir).startswith(str(config.project_root))

    def test_models_dir_under_project(self) -> None:
        assert str(config.models_dir).startswith(str(config.project_root))

    def test_db_path_ends_with_sqlite(self) -> None:
        assert config.db_path.suffix == ".db"

    def test_app_name(self) -> None:
        assert config.app_name == "ReminisceCV"

    def test_app_version(self) -> None:
        assert isinstance(config.app_version, str)
        assert len(config.app_version) > 0

    # ── Camera defaults ──

    def test_camera_index_default(self) -> None:
        assert config.camera_index == 0

    def test_frame_width_default(self) -> None:
        assert config.frame_width == 640

    def test_frame_height_default(self) -> None:
        assert config.frame_height == 480

    # ── CLIP defaults ──

    def test_clip_model_name(self) -> None:
        assert "clip" in config.clip_model_name.lower()

    def test_clip_embedding_dim(self) -> None:
        assert config.clip_embedding_dim == 512

    def test_object_threshold_range(self) -> None:
        assert 0.0 < config.object_similarity_threshold < 1.0

    # ── Face defaults ──

    def test_face_model_name(self) -> None:
        assert isinstance(config.face_model_name, str)

    def test_face_similarity_threshold_range(self) -> None:
        assert 0.0 < config.face_similarity_threshold < 1.0

    def test_face_embedding_dim(self) -> None:
        assert config.face_embedding_dim == 512

    # ── Recognition defaults ──

    def test_debounce_frames_positive(self) -> None:
        assert config.debounce_frames > 0

    def test_recognition_stability_positive(self) -> None:
        assert config.recognition_stability_sec > 0.0

    def test_speech_cooldown_positive(self) -> None:
        assert config.speech_cooldown_sec > 0.0

    # ── TTS defaults ──

    def test_tts_rate_reasonable(self) -> None:
        assert 50 <= config.tts_rate <= 400

    def test_tts_volume_range(self) -> None:
        assert 0.0 <= config.tts_volume <= 1.0


class TestConfigEnvironmentOverrides:
    """Verify that environment variables override defaults."""

    def test_camera_index_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RCV_CAMERA_INDEX", "2")
        c = Config()
        assert c.camera_index == 2

    def test_frame_width_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RCV_FRAME_WIDTH", "1280")
        c = Config()
        assert c.frame_width == 1280

    def test_object_threshold_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RCV_OBJECT_THRESHOLD", "0.35")
        c = Config()
        assert c.object_similarity_threshold == pytest.approx(0.35)

    def test_tts_rate_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RCV_TTS_RATE", "200")
        c = Config()
        assert c.tts_rate == 200

    def test_db_path_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RCV_DB_PATH", "/tmp/test.db")
        c = Config()
        assert str(c.db_path) == str(Path("/tmp/test.db"))

    def test_speech_cooldown_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RCV_SPEECH_COOLDOWN", "60.0")
        c = Config()
        assert c.speech_cooldown_sec == pytest.approx(60.0)


class TestConfigMethods:
    """Verify Config helper methods."""

    def test_summary_returns_string(self) -> None:
        s = config.summary()
        assert isinstance(s, str)
        assert "app_name" in s
        assert "ReminisceCV" in s

    def test_summary_contains_all_sections(self) -> None:
        s = config.summary()
        assert "camera_index" in s
        assert "clip_model_name" in s
        assert "face_model_name" in s
        assert "debounce_frames" in s
        assert "tts_rate" in s

    def test_ensure_directories_creates_dirs(self, tmp_path: Path) -> None:
        c = Config()
        c.data_dir = tmp_path / "test_data"
        c.models_dir = tmp_path / "test_models"
        c.ensure_directories()
        assert c.data_dir.exists()
        assert c.models_dir.exists()
