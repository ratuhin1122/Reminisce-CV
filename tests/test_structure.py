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
    from app.config import config

    assert isinstance(config.project_root, Path)
    assert isinstance(config.data_dir, Path)
    assert isinstance(config.models_dir, Path)
    assert isinstance(config.db_path, Path)


def test_config_paths_are_absolute() -> None:
    """Config paths should be absolute, not relative."""
    from app.config import config

    assert config.project_root.is_absolute()
    assert config.data_dir.is_absolute()
    assert config.models_dir.is_absolute()
    assert config.db_path.is_absolute()


def test_main_function_exists() -> None:
    """The main module should have a callable main() function."""
    from app.main import main
    assert callable(main)
