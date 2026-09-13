"""
ReminisceCV — Central Configuration
====================================

All tuneable parameters live here as a single ``Config`` dataclass.
Modules import the singleton ``config`` instance:

    from app.config import config
    threshold = config.object_similarity_threshold

Design rules
------------
- No secrets or personal data in this file.
- Every field has a safe default.
- Environment variables (prefixed ``RCV_``) can override defaults.
- Paths are resolved relative to the project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    """Read an environment variable with a fallback default."""
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback default."""
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a fallback default."""
    return float(os.environ.get(name, str(default)))


# ── Project-level constants (not configurable) ────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Centralized configuration for the ReminisceCV application.

    Every subsystem reads its parameters from this single object.
    Values can be overridden via environment variables prefixed
    with ``RCV_`` (e.g. ``RCV_CAMERA_INDEX=1``).

    Attributes
    ----------
    Paths
        project_root, data_dir, models_dir, db_path
    Application
        app_name, app_version
    Camera
        camera_index, frame_width, frame_height
    Vision / CLIP
        clip_model_name, clip_pretrained, clip_embedding_dim,
        object_similarity_threshold
    Face Recognition
        face_model_name, face_det_threshold,
        face_similarity_threshold, face_embedding_dim
    Recognition Engine
        debounce_frames, recognition_stability_sec,
        speech_cooldown_sec
    Text-to-Speech
        tts_rate, tts_volume
    """

    # ── Paths ─────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    db_path: Path = field(
        default_factory=lambda: Path(
            _env("RCV_DB_PATH", str(PROJECT_ROOT / "data" / "reminiscecv.db"))
        )
    )

    # ── Application ───────────────────────────────────────────
    app_name: str = "ReminisceCV"
    app_version: str = "0.1.0"

    # ── Camera ────────────────────────────────────────────────
    camera_index: int = field(
        default_factory=lambda: _env_int("RCV_CAMERA_INDEX", 0)
    )
    frame_width: int = field(
        default_factory=lambda: _env_int("RCV_FRAME_WIDTH", 640)
    )
    frame_height: int = field(
        default_factory=lambda: _env_int("RCV_FRAME_HEIGHT", 480)
    )

    # ── Vision / CLIP Embeddings ──────────────────────────────
    clip_model_name: str = field(
        default_factory=lambda: _env("RCV_CLIP_MODEL", "openai/clip-vit-base-patch32")
    )
    clip_embedding_dim: int = 512
    object_similarity_threshold: float = field(
        default_factory=lambda: _env_float("RCV_OBJECT_THRESHOLD", 0.25)
    )

    # ── Face Recognition ──────────────────────────────────────
    face_model_name: str = field(
        default_factory=lambda: _env("RCV_FACE_MODEL", "buffalo_l")
    )
    face_det_threshold: float = 0.5
    face_similarity_threshold: float = field(
        default_factory=lambda: _env_float("RCV_FACE_THRESHOLD", 0.4)
    )
    face_embedding_dim: int = 512

    # ── Recognition Engine ────────────────────────────────────
    debounce_frames: int = field(
        default_factory=lambda: _env_int("RCV_DEBOUNCE_FRAMES", 3)
    )
    recognition_stability_sec: float = field(
        default_factory=lambda: _env_float("RCV_STABILITY_SEC", 1.0)
    )
    speech_cooldown_sec: float = field(
        default_factory=lambda: _env_float("RCV_SPEECH_COOLDOWN", 30.0)
    )

    # ── Text-to-Speech ────────────────────────────────────────
    tts_rate: int = field(
        default_factory=lambda: _env_int("RCV_TTS_RATE", 160)
    )
    tts_volume: float = field(
        default_factory=lambda: _env_float("RCV_TTS_VOLUME", 0.9)
    )

    def ensure_directories(self) -> None:
        """Create required directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """Return a human-readable summary of the current configuration."""
        lines = [
            f"  app_name             = {self.app_name}",
            f"  app_version          = {self.app_version}",
            f"  project_root         = {self.project_root}",
            f"  data_dir             = {self.data_dir}",
            f"  models_dir           = {self.models_dir}",
            f"  db_path              = {self.db_path}",
            "",
            f"  camera_index         = {self.camera_index}",
            f"  frame_width          = {self.frame_width}",
            f"  frame_height         = {self.frame_height}",
            "",
            f"  clip_model_name      = {self.clip_model_name}",
            f"  clip_embedding_dim   = {self.clip_embedding_dim}",
            f"  object_threshold     = {self.object_similarity_threshold}",
            "",
            f"  face_model_name      = {self.face_model_name}",
            f"  face_det_threshold   = {self.face_det_threshold}",
            f"  face_similarity_thr  = {self.face_similarity_threshold}",
            f"  face_embedding_dim   = {self.face_embedding_dim}",
            "",
            f"  debounce_frames      = {self.debounce_frames}",
            f"  stability_sec        = {self.recognition_stability_sec}",
            f"  speech_cooldown_sec  = {self.speech_cooldown_sec}",
            "",
            f"  tts_rate             = {self.tts_rate}",
            f"  tts_volume           = {self.tts_volume}",
        ]
        return "\n".join(lines)


# ── Singleton instance ────────────────────────────────────────
# Import this throughout the application:
#     from app.config import config
config = Config()


# ── Backward-compatible aliases (used by main.py Stage 1) ────
APP_NAME = config.app_name
DATA_DIR = config.data_dir
MODELS_DIR = config.models_dir
DB_PATH = config.db_path
