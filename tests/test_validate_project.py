from __future__ import annotations

from pathlib import Path

from scripts.validate_project import should_exclude_path


def test_validate_project_excludes_generated_and_environment_directories():
    paths = [
        Path(".git") / "objects" / "abc.py",
        Path("review_packages") / "pkg" / "source.py",
        Path("reports") / "experiments" / "report.py",
        Path("storage") / "scratch.py",
        Path(".venv") / "Lib" / "site-packages" / "pkg.py",
        Path("__pycache__") / "module.py",
        Path(".streamlit") / "secrets.toml",
    ]

    assert all(should_exclude_path(path) for path in paths)


def test_validate_project_keeps_project_source_paths():
    paths = [
        Path("research") / "experiment_store.py",
        Path("app.py"),
        Path("tests") / "test_experiment_store.py",
    ]

    assert not any(should_exclude_path(path) for path in paths)
