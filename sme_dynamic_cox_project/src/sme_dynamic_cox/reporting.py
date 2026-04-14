from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_metrics(metrics: Dict, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_dataframe(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    ensure_output_dir(output_dir)
    df.to_csv(output_dir / name, index=False, encoding="utf-8-sig")


def plot_top_coefficients(model_summary: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    ensure_output_dir(output_dir)
    if model_summary.empty:
        return

    summary = model_summary.copy()
    if "coef" not in summary.columns:
        return

    summary["abs_coef"] = summary["coef"].abs()
    plot_df = summary.sort_values("abs_coef", ascending=False).head(top_n).sort_values("coef")
    if plot_df.empty:
        return

    plt.figure(figsize=(8, max(4, len(plot_df) * 0.4)))
    plt.barh(plot_df.index.astype(str), plot_df["coef"])
    plt.xlabel("Coefficient (Dynamic Cox)")
    plt.ylabel("Feature")
    plt.title("Top Feature Effects on Default Hazard")
    plt.tight_layout()
    plt.savefig(output_dir / "top_coefficients.png", dpi=200)
    plt.close()


def build_readable_summary(metrics: Dict) -> str:
    lines = [
        "动态Cox模型实验摘要",
        f"- 贷款源文件: {metrics.get('loan_file_name', 'N/A')}",
        f"- 数据切分时间点: {metrics.get('split_cutoff_date', 'N/A')}",
        f"- 训练集贷款数: {metrics.get('n_train_loans', 'N/A')}",
        f"- 测试集贷款数: {metrics.get('n_test_loans', 'N/A')}",
        f"- 训练集事件数: {metrics.get('train_events', 'N/A')}",
        f"- 测试集事件数: {metrics.get('test_events', 'N/A')}",
        f"- 训练集C-index: {metrics.get('train_c_index', float('nan')):.4f}",
        f"- 测试集C-index: {metrics.get('test_c_index', float('nan')):.4f}",
        f"- 动态预警AUC: {metrics.get('warning_roc_auc', float('nan')):.4f}",
        f"- 动态预警F1: {metrics.get('warning_f1', float('nan')):.4f}",
    ]
    return "\n".join(lines)

