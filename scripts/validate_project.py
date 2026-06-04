from __future__ import annotations

import compileall
import importlib
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    "review_packages",
    "storage",
    ".streamlit",
}
EXCLUDED_RELATIVE_DIRS = {
    Path("reports") / "experiments",
    Path("reports") / "data_quality",
}
COMPILE_EXCLUDE_REGEX = re.compile(
    r"(^|[\\/])("
    + "|".join(re.escape(name) for name in sorted(EXCLUDED_DIR_NAMES))
    + r")([\\/]|$)|(^|[\\/])reports[\\/](experiments|data_quality)([\\/]|$)"
)

REQUIRED_DIRS = [
    "backtest",
    "data",
    "data_quality",
    "execution",
    "factor",
    "market",
    "optimizer",
    "portfolio",
    "ranking",
    "report",
    "research",
    "reports/experiments",
    "reports/research",
    "reports/candidates",
    "reports/data_quality",
    "stability",
    "strategy",
    "candidate",
    "tests",
    "workflow",
    "storage",
]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "app.py",
    "config.py",
    "backtest/backtest_engine.py",
    "data/database.py",
    "data_quality/models.py",
    "data_quality/calendar.py",
    "data_quality/validators.py",
    "data_quality/universe.py",
    "data_quality/adjustment.py",
    "data_quality/audit.py",
    "data_quality/data_health_service.py",
    "data_quality/report_exporter.py",
    "data_quality/ui_helpers.py",
    "execution/cost_model.py",
    "execution/slippage_model.py",
    "execution/order_model.py",
    "factor/factor_engine.py",
    "market/index_data.py",
    "market/regime_engine.py",
    "optimizer/parameter_optimizer.py",
    "portfolio/position_sizer.py",
    "portfolio/portfolio_engine.py",
    "portfolio/portfolio_metrics.py",
    "portfolio/strategy_ensemble.py",
    "ranking/leaderboard.py",
    "report/charts.py",
    "research/experiment_store.py",
    "research/experiment_models.py",
    "research/experiment_service.py",
    "research/experiment_compare.py",
    "research/report_exporter.py",
    "research/ui_helpers.py",
    "candidate/candidate_models.py",
    "candidate/candidate_decision.py",
    "candidate/candidate_store.py",
    "candidate/candidate_revalidator.py",
    "candidate/candidate_report.py",
    "candidate/candidate_ui_helpers.py",
    "stability/splitter.py",
    "stability/walk_forward.py",
    "stability/parameter_sensitivity.py",
    "stability/overfit_detector.py",
    "stability/stability_metrics.py",
    "stability/stability_service.py",
    "workflow/research_models.py",
    "workflow/research_templates.py",
    "workflow/research_scorecard.py",
    "workflow/research_store.py",
    "workflow/research_workflow.py",
    "workflow/research_report.py",
    "strategy/metadata.py",
    "tests/test_cost_model.py",
    "tests/test_slippage_model.py",
    "tests/test_order_model.py",
    "tests/test_position_sizer.py",
    "tests/test_portfolio_engine.py",
    "tests/test_strategy_ensemble.py",
    "tests/test_portfolio_metrics.py",
    "tests/test_market_regime_v3.py",
    "tests/test_backtest_engine.py",
    "tests/test_market_regime.py",
    "tests/test_optimizer.py",
    "tests/test_leaderboard.py",
    "tests/test_experiment_store.py",
    "tests/test_experiment_service.py",
    "tests/test_experiment_compare.py",
    "tests/test_report_exporter.py",
    "tests/test_experiment_ui_helpers.py",
    "tests/test_validate_project.py",
    "tests/test_splitter.py",
    "tests/test_walk_forward.py",
    "tests/test_parameter_sensitivity.py",
    "tests/test_overfit_detector.py",
    "tests/test_stability_metrics.py",
    "tests/test_stability_service.py",
    "tests/test_stability_report_export.py",
    "tests/test_stability_params.py",
    "tests/test_research_models.py",
    "tests/test_research_templates.py",
    "tests/test_research_scorecard.py",
    "tests/test_research_store.py",
    "tests/test_research_report.py",
    "tests/test_research_workflow.py",
    "tests/test_candidate_models.py",
    "tests/test_candidate_decision.py",
    "tests/test_candidate_store.py",
    "tests/test_candidate_revalidator.py",
    "tests/test_candidate_report.py",
    "tests/test_candidate_ui_helpers.py",
    "tests/test_data_quality_calendar.py",
    "tests/test_data_quality_validators.py",
    "tests/test_adjustment.py",
    "tests/test_universe_builder.py",
    "tests/test_data_quality_audit.py",
    "tests/test_data_health_service.py",
    "tests/test_data_quality_report.py",
    "pytest.ini",
]


def main() -> int:
    print("A-Stock Quant Lab local validation")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Excluded directories: {', '.join(sorted(EXCLUDED_DIR_NAMES | {'reports/experiments', 'reports/data_quality'}))}")

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
    print("Command: python -m compileall <project> with generated/runtime directories excluded")
    ok = compileall.compile_dir(PROJECT_ROOT, quiet=1, rx=COMPILE_EXCLUDE_REGEX)
    if not ok:
        raise RuntimeError("compileall reported failures")


def check_app_import() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    app = importlib.import_module("app")
    if not hasattr(app, "main"):
        raise AttributeError("app.py does not expose main()")


def run_command(command: list[str]) -> None:
    print(f"Command: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def should_exclude_path(path: str | Path) -> bool:
    candidate = Path(path)
    parts = set(candidate.parts)
    if parts & EXCLUDED_DIR_NAMES:
        return True
    normalized = Path(*candidate.parts) if candidate.parts else candidate
    for excluded in EXCLUDED_RELATIVE_DIRS:
        if normalized == excluded or excluded in _parents(normalized):
            return True
    return False


def _parents(path: Path) -> set[Path]:
    parents: set[Path] = set()
    current = path
    while current.parent != current:
        current = current.parent
        parents.add(current)
        if len(current.parts) <= 1:
            break
    return parents


if __name__ == "__main__":
    raise SystemExit(main())
