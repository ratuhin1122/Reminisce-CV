"""
ReminisceCV — Central Configuration
====================================

All tuneable parameters live here. Values are organized by subsystem
so that each module imports only what it needs.

Design rules
------------
- No secrets or personal data in this file.
- Paths are relative to the project root.
- AI-model and threshold values will be added in later stages.
"""

from pathlib import Path

# ── Project Paths ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = PROJECT_ROOT / "data" / "reminiscecv.db"

# ── Application ───────────────────────────────────────────────
APP_NAME = "ReminisceCV"
APP_VERSION = "0.1.0"

# ── Database (Stage 2) ───────────────────────────────────────
# DB_PATH is defined above. Schema details added in Stage 2.

# ── Vision / CLIP Embeddings (Stage 3) ────────────────────────
# CLIP_MODEL_NAME = "ViT-B-32"
# CLIP_PRETRAINED = "laion2b_s34b_b79k"
# CLIP_EMBEDDING_DIM = 512
# OBJECT_SIMILARITY_THRESHOLD = 0.25

# ── Face Recognition (Stage 4) ───────────────────────────────
# FACE_MODEL_NAME = "buffalo_l"
# FACE_DET_THRESHOLD = 0.5
# FACE_SIMILARITY_THRESHOLD = 0.4
# FACE_EMBEDDING_DIM = 512

# ── Camera / Real-time Loop (Stage 5) ────────────────────────
# CAMERA_INDEX = 0
# TARGET_FPS = 15
# FRAME_WIDTH = 640
# FRAME_HEIGHT = 480

# ── Recognition Engine (Stage 5) ─────────────────────────────
# DEBOUNCE_FRAMES = 3          # consecutive frames before confirming
# SPEECH_COOLDOWN_SEC = 30.0   # seconds before re-announcing same entity

# ── Text-to-Speech (Stage 6) ─────────────────────────────────
# TTS_RATE = 160               # words per minute
# TTS_VOLUME = 0.9
