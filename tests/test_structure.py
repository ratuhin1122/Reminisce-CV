"""
tests.test_structure — Verify project structure integrity
==========================================================

Smoke tests that confirm all packages are importable and the
project directory layout is correct.
"""

import importlib

import pytest


# Every sub-package that must be importable
EXPECTED_PACKAGES = [
    "app",
    "app.config",
    "app.main",
    "app.database",
    "app.vision",
    "app.recognition",
    "app.memory",
    "app.speech",
    "app.ui",
]


@pytest.mark.parametrize("package", EXPECTED_PACKAGES)
def test_package_importable(package: str) -> None:
    """Each sub-package should import without errors."""
    mod = importlib.import_module(package)
    assert mod is not None


def test_app_version_exists() -> None:
    """The app package should expose a version string."""
    from app import __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_app_project_name() -> None:
    """The app package should expose the project name."""
    from app import __project__
    assert __project__ == "ReminisceCV"


def test_config_paths_are_pathlib() -> None:
    """Config paths should be pathlib.Path objects."""
    from pathlib import Path
    from app.config import PROJECT_ROOT, DATA_DIR, MODELS_DIR, DB_PATH

    assert isinstance(PROJECT_ROOT, Path)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(MODELS_DIR, Path)
    assert isinstance(DB_PATH, Path)


def test_config_paths_are_absolute() -> None:
    """Config paths should be absolute, not relative."""
    from app.config import PROJECT_ROOT, DATA_DIR, MODELS_DIR, DB_PATH

    assert PROJECT_ROOT.is_absolute()
    assert DATA_DIR.is_absolute()
    assert MODELS_DIR.is_absolute()
    assert DB_PATH.is_absolute()


def test_main_function_exists() -> None:
    """The main module should have a callable main() function."""
    from app.main import main
    assert callable(main)
