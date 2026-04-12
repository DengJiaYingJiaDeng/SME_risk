from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.sme_dynamic_cox.config import ExperimentConfig
from src.sme_dynamic_cox.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SME Dynamic Cox experiment for loan approval and default warning.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"E:\my model\SME_model\SME - Synthetic UK Businesses financial statistics",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"E:\my model\SME_model\sme_dynamic_cox_project\outputs1",
        help="Path to save experiment outputs.",
    )
    parser.add_argument("--snapshot-date", type=str, default="2024-02-29", help="Right-censor snapshot date.")
    parser.add_argument("--warning-horizon-days", type=int, default=90, help="Dynamic warning horizon in days.")
    parser.add_argument("--decision-horizon-days", type=int, default=180, help="Loan decision horizon in days.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Approximate test set ratio.")
    parser.add_argument("--penalizer", type=float, default=0.25, help="Cox regularization strength.")
    parser.add_argument("--l1-ratio", type=float, default=0.0, help="Cox elastic-net l1 ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        snapshot_date=args.snapshot_date,
        warning_horizon_days=args.warning_horizon_days,
        decision_horizon_days=args.decision_horizon_days,
        test_size=args.test_size,
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
    )

    metrics = run_experiment(config)
    print("=== Experiment Completed ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

