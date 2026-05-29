from __future__ import annotations

import compileall
import importlib
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "backtest",
    "data",
    "factor",
    "market",
    "optimizer",
    "ranking",
    "report",
    "strategy",
    "tests",
    "storage",
]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "app.py",
    "config.py",
    "backtest/backtest_engine.py",
    "data/database.py",
    "factor/factor_engine.py",
    "market/regime_engine.py",
    "optimizer/parameter_optimizer.py",
    "ranking/leaderboard.py",
    "report/charts.py",
    "strategy/metadata.py",
    "tests/test_backtest_engine.py",
    "tests/test_market_regime.py",
    "tests/test_optimizer.py",
    "tests/test_leaderboard.py",
]


def main() -> int:
    print("A-Stock Quant Lab local validation")
    print(f"Project root: {PROJECT_ROOT}")

    checks = [
        ("directories", check_directories),
        ("files", check_files),
        ("pytest", run_pytest),
        ("compileall", run_compileall),
        ("app import", check_app_import),
    ]

    failures: list[str] = []
    for name, check in checks:
        print(f"\n[{name}]")
        try:
            check()
        except Exception as exc:
            failures.append(name)
            print(f"FAIL: {exc}")
        else:
            print("PASS")

    print("\nValidation summary")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1

    print("PASSED: all checks completed")
    return 0


def check_directories() -> None:
    missing = [path for path in REQUIRED_DIRS if not (PROJECT_ROOT / path).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing directories: {missing}")


def check_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")


def run_pytest() -> None:
    run_command([sys.executable, "-m", "pytest", "-q"])


def run_compileall() -> None:
    ok = compileall.compile_dir(PROJECT_ROOT, quiet=1)
    if not ok:
        raise RuntimeError("compileall reported failures")


def check_app_import() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    app = importlib.import_module("app")
    if not hasattr(app, "main"):
        raise AttributeError("app.py does not expose main()")


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


if __name__ == "__main__":
    raise SystemExit(main())
