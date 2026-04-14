from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

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
    if model_summary.empty or "coef" not in model_summary.columns:
        return

    summary = model_summary.copy()
    summary["abs_coef"] = summary["coef"].abs()
    plot_df = summary.sort_values("abs_coef", ascending=False).head(top_n).sort_values("coef")
    if plot_df.empty:
        return

    plt.figure(figsize=(8, max(4, len(plot_df) * 0.45)))
    plt.barh(plot_df.index.astype(str), plot_df["coef"], color="#2a9d8f")
    plt.axvline(0.0, color="gray", linewidth=1)
    plt.xlabel("Coefficient (Dynamic Cox)")
    plt.ylabel("Feature")
    plt.title("Top Feature Effects on Default Hazard")
    plt.tight_layout()
    plt.savefig(output_dir / "top_coefficients.png", dpi=220)
    plt.close()


def plot_time_dependent_curves(time_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    if time_df.empty:
        return

    plot_df = time_df.sort_values("horizon_days")
    x = plot_df["horizon_days"]
    plt.figure(figsize=(8.5, 5.2))
    if "time_dependent_auc" in plot_df.columns:
        plt.plot(x, plot_df["time_dependent_auc"], marker="o", label="Time-dependent AUC")
    if "ks_statistic" in plot_df.columns:
        plt.plot(x, plot_df["ks_statistic"], marker="s", label="KS Statistic")
    if "brier_score" in plot_df.columns:
        plt.plot(x, plot_df["brier_score"], marker="^", label="Brier Score")
    if "brier_skill_score" in plot_df.columns:
        plt.plot(x, plot_df["brier_skill_score"], marker="d", label="Brier Skill")
    plt.xlabel("Prediction Horizon (days)")
    plt.ylabel("Metric Value")
    plt.title("Time-dependent Metrics Across Horizons")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "time_dependent_metrics_curve.png", dpi=220)
    plt.close()


def plot_stratified_survival(
    test_df: pd.DataFrame,
    score_col: str,
    duration_col: str,
    event_col: str,
    split_threshold: float,
    output_dir: Path,
) -> None:
    ensure_output_dir(output_dir)
    if test_df.empty:
        return

    df = test_df.copy()
    df["risk_group"] = np.where(df[score_col] >= split_threshold, "High-Risk", "Prime")
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(8.5, 5.2))
    for group, color in [("High-Risk", "#e63946"), ("Prime", "#2a9d8f")]:
        grp = df[df["risk_group"] == group]
        if grp.empty:
            continue
        kmf.fit(
            durations=pd.to_numeric(grp[duration_col], errors="coerce").fillna(1).clip(lower=1),
            event_observed=pd.to_numeric(grp[event_col], errors="coerce").fillna(0).astype(int),
            label=group,
        )
        surv = kmf.survival_function_.copy()
        surv.index = surv.index / 30.0
        plt.step(surv.index, surv.iloc[:, 0], where="post", linewidth=2.2, color=color, label=group)

    plt.xlabel("Months Since Origination")
    plt.ylabel("Survival Probability")
    plt.title("Stratified Survival Curves (High-Risk vs Prime)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stratified_survival_curve.png", dpi=220)
    plt.close()


def plot_tradeoff_curve(
    tradeoff_df: pd.DataFrame,
    output_dir: Path,
    bad_rate_target: float = 0.03,
) -> None:
    ensure_output_dir(output_dir)
    if tradeoff_df.empty:
        return

    df = tradeoff_df.sort_values("approval_rate").copy()
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(df["approval_rate"], df["bad_debt_rate"], linewidth=2.0, color="#264653")
    plt.axhline(bad_rate_target, linestyle="--", color="#e76f51", label=f"Bad debt target = {bad_rate_target:.0%}")

    feasible = df[df["bad_debt_rate"] <= bad_rate_target]
    if not feasible.empty:
        best = feasible.sort_values("approval_rate", ascending=False).iloc[0]
        plt.scatter([best["approval_rate"]], [best["bad_debt_rate"]], color="#f4a261", s=80, zorder=3)
        plt.annotate(
            f"max approve={best['approval_rate']:.1%}",
            (best["approval_rate"], best["bad_debt_rate"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    plt.xlabel("Approval Rate")
    plt.ylabel("Bad Debt Rate (Approved Only)")
    plt.title("Strategy Trade-off Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_tradeoff_curve.png", dpi=220)
    plt.close()


def plot_local_waterfall(
    contribution_series: pd.Series,
    output_path: Path,
    title: str,
    top_n: int = 10,
) -> None:
    if contribution_series.empty:
        return
    contrib = contribution_series.sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)
    labels = list(contrib.index)
    values = contrib.values.astype(float)
    cumulative = np.cumsum(np.r_[0.0, values[:-1]])

    colors = np.where(values >= 0, "#e76f51", "#2a9d8f")
    plt.figure(figsize=(9.5, 5.4))
    for i, (base, val, color) in enumerate(zip(cumulative, values, colors)):
        plt.bar(i, val, bottom=base, color=color, width=0.7)
    plt.plot(
        range(len(values) + 1),
        np.cumsum(np.r_[0.0, values]),
        color="#264653",
        marker="o",
        linewidth=1.5,
    )
    plt.axhline(0.0, color="gray", linewidth=1)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("Contribution to Linear Predictor")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_pdp_curves(
    pdp_frame: pd.DataFrame,
    output_dir: Path,
    feature_order: Iterable[str],
) -> None:
    ensure_output_dir(output_dir)
    if pdp_frame.empty:
        return

    features = [f for f in feature_order if f in pdp_frame["feature"].unique()]
    if not features:
        features = list(pdp_frame["feature"].drop_duplicates().head(2))
    if not features:
        return

    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.6), squeeze=False)
    for idx, feat in enumerate(features):
        ax = axes[0, idx]
        d = pdp_frame[pdp_frame["feature"] == feat].sort_values("feature_value")
        if d.empty:
            continue
        ax.plot(d["feature_value"], d["mean_partial_hazard"], color="#457b9d", linewidth=2.0)
        ax.set_title(f"PDP: {feat}")
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Mean Predicted Hazard")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "partial_dependence_plots.png", dpi=220)
    plt.close(fig)


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
        f"- 90天预警AUC: {metrics.get('warning_roc_auc', float('nan')):.4f}",
        f"- 90天预警F1: {metrics.get('warning_f1', float('nan')):.4f}",
        f"- 90天KS: {metrics.get('warning_ks_90d', float('nan')):.4f}",
        f"- 90天Brier: {metrics.get('warning_brier_90d', float('nan')):.4f}",
        f"- 90天Brier Skill: {metrics.get('warning_brier_skill_90d', float('nan')):.4f}",
    ]
    return "\n".join(lines)
