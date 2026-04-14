from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.sme_dynamic_cox.config import ExperimentConfig
from src.sme_dynamic_cox.pipeline import run_experiment


def _safe_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _with_derived(metrics: Dict) -> Dict:
    out = dict(metrics)
    train_events = _safe_float(out.get("train_events"))
    n_features = max(_safe_float(out.get("n_features")), 1.0)
    out["train_epv_proxy"] = train_events / n_features
    return out


def _build_comparison_table(before: Dict, after: Dict) -> pd.DataFrame:
    metrics_order: List[str] = [
        "train_events",
        "test_events",
        "train_epv_proxy",
        "train_c_index",
        "test_c_index",
        "warning_precision",
        "warning_recall",
        "warning_f1",
        "warning_avg_precision",
        "warning_roc_auc",
    ]
    higher_better = set(metrics_order)

    rows = []
    for metric in metrics_order:
        b = _safe_float(before.get(metric))
        a = _safe_float(after.get(metric))
        delta = a - b
        improved = delta > 0 if metric in higher_better else None
        rows.append(
            {
                "metric": metric,
                "before_original": b,
                "after_augmented": a,
                "delta_after_minus_before": delta,
                "improved": improved,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data_dir = Path(r"E:\my model\SME_model\SME - Synthetic UK Businesses financial statistics")
    output_root = Path(r"E:\my model\SME_model\sme_dynamic_cox_project\outputs\comparison")
    before_dir = output_root / "before_original"
    after_dir = output_root / "after_augmented"
    output_root.mkdir(parents=True, exist_ok=True)

    before_cfg = ExperimentConfig(
        data_dir=data_dir,
        output_dir=before_dir,
        loan_file_name="Loan.csv",
        snapshot_date="2024-02-29",
        warning_horizon_days=90,
        test_size=0.3,
        penalizer=0.25,
        l1_ratio=0.0,
    )
    after_cfg = ExperimentConfig(
        data_dir=data_dir,
        output_dir=after_dir,
        loan_file_name="Loan_Augmented_RealSignals.csv",
        snapshot_date="2024-02-29",
        warning_horizon_days=90,
        test_size=0.3,
        penalizer=0.25,
        l1_ratio=0.0,
    )

    before_metrics = _with_derived(run_experiment(before_cfg))
    after_metrics = _with_derived(run_experiment(after_cfg))

    comparison = _build_comparison_table(before_metrics, after_metrics)
    comparison_csv = output_root / "before_vs_after_metrics.csv"
    comparison_md = output_root / "before_vs_after_metrics.md"
    comparison.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    md_lines = [
        "| metric | before_original | after_augmented | delta_after_minus_before | improved |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in comparison.itertuples(index=False):
        md_lines.append(
            f"| {row.metric} | {row.before_original:.6f} | {row.after_augmented:.6f} | "
            f"{row.delta_after_minus_before:.6f} | {row.improved} |"
        )
    comparison_md.write_text("\n".join(md_lines), encoding="utf-8")

    summary = {
        "before_original": before_metrics,
        "after_augmented": after_metrics,
        "comparison_table_file": str(comparison_csv),
    }
    (output_root / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== Comparison Completed ===")
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison to: {comparison_csv}")


if __name__ == "__main__":
    main()
