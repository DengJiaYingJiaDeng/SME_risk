from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data_io import load_all_tables
from .evaluation import add_future_event_label, build_origination_view, choose_time_split, classification_metrics
from .feature_builder import build_modeling_dataset, select_feature_columns
from .model import DynamicCoxRiskModel
from .policy import (
    apply_approval_policy,
    apply_warning_policy,
    build_industry_approval_policy,
    build_industry_warning_policy,
)
from .reporting import build_readable_summary, plot_top_coefficients, save_dataframe, save_metrics


def _make_industry_decision_summary(decision_df: pd.DataFrame) -> pd.DataFrame:
    def _approved_default_rate(group: pd.DataFrame) -> float:
        approved = group[group["loan_decision"] == "Approve"]
        if approved.empty:
            return float("nan")
        return float(approved["event"].mean())

    grouped = decision_df.groupby("industry_group")
    summary = grouped.agg(
        n_loans=("loan_key", "count"),
        observed_default_rate=("event", "mean"),
        avg_risk_score=("risk_score", "mean"),
        approval_rate=("loan_decision", lambda x: (x == "Approve").mean()),
    )
    summary["approved_default_rate"] = grouped.apply(_approved_default_rate)
    return summary.reset_index().sort_values(["observed_default_rate", "n_loans"], ascending=[False, False])


def run_experiment(config: ExperimentConfig) -> Dict:
    tables = load_all_tables(config.data_dir)
    prepared = build_modeling_dataset(tables, config)
    loan_master = prepared.loan_master.copy()
    interval_data = prepared.interval_data.copy()

    feature_cols = select_feature_columns(interval_data, config)
    if not feature_cols:
        raise RuntimeError("No valid numeric features selected for the Cox model.")

    cutoff, train_loans, test_loans = choose_time_split(
        loan_master,
        test_size=config.test_size,
        min_train_events=config.min_train_events,
        min_test_events=config.min_test_events,
    )

    train_ids = set(train_loans["loan_key"].astype(str))
    test_ids = set(test_loans["loan_key"].astype(str))
    train_intervals = interval_data[interval_data["loan_key"].astype(str).isin(train_ids)].copy()
    test_intervals = interval_data[interval_data["loan_key"].astype(str).isin(test_ids)].copy()

    model = DynamicCoxRiskModel(penalizer=config.penalizer, l1_ratio=config.l1_ratio, strata_col="industry_group")
    model.fit(train_intervals, feature_cols=feature_cols)

    # Loan-level decision view from origination time.
    origination_df = build_origination_view(interval_data, loan_master)
    origination_df["risk_score"] = model.predict_partial_hazard(origination_df)
    train_orig = origination_df[origination_df["loan_key"].astype(str).isin(train_ids)].copy()
    test_orig = origination_df[origination_df["loan_key"].astype(str).isin(test_ids)].copy()

    train_c = model.concordance_on_loans(train_orig, score_col="risk_score")
    test_c = model.concordance_on_loans(test_orig, score_col="risk_score")

    approval_policy = build_industry_approval_policy(
        train_orig,
        industry_col="industry_group",
        score_col="risk_score",
        target_col="event",
        min_samples_per_industry=config.min_samples_per_industry,
        gain=config.loan_approval_gain,
        loss=config.loan_default_loss,
    )
    loan_decisions = apply_approval_policy(
        origination_df,
        policy_df=approval_policy,
        industry_col="industry_group",
        score_col="risk_score",
    )
    industry_decision_summary = _make_industry_decision_summary(loan_decisions)

    # Dynamic monitoring + warning labels.
    monitored_df = add_future_event_label(interval_data, horizon_days=config.warning_horizon_days)
    monitored_df = monitored_df[monitored_df["monitoring_row"] == 1].copy()
    monitored_df[f"p_default_{config.warning_horizon_days}d"] = model.predict_default_probability(
        monitored_df, horizon_days=config.warning_horizon_days
    )
    prob_col = f"p_default_{config.warning_horizon_days}d"

    train_monitor = monitored_df[monitored_df["loan_key"].astype(str).isin(train_ids)].copy()
    test_monitor = monitored_df[monitored_df["loan_key"].astype(str).isin(test_ids)].copy()

    warning_policy = build_industry_warning_policy(
        train_monitor,
        industry_col="industry_group",
        prob_col=prob_col,
        target_col="future_event_label",
        min_samples_per_industry=30,
    )
    warning_scored = apply_warning_policy(
        monitored_df,
        policy_df=warning_policy,
        industry_col="industry_group",
        prob_col=prob_col,
    )

    global_warning_threshold = float(warning_policy["warning_threshold"].median()) if not warning_policy.empty else 0.2
    test_warning = warning_scored[warning_scored["loan_key"].astype(str).isin(test_ids)].copy()
    warn_metrics = classification_metrics(
        y_true=test_warning["future_event_label"],
        y_score=test_warning[prob_col],
        threshold=global_warning_threshold,
    )

    latest_warning = (
        warning_scored.sort_values(["loan_key", "month_end", prob_col]).groupby("loan_key", as_index=False).tail(1)
    )
    latest_warning = latest_warning.sort_values(["risk_level", prob_col], ascending=[True, False]).reset_index(drop=True)

    # Outputs
    output_dir = Path(config.output_dir)
    save_dataframe(loan_master, output_dir, "loan_master_cleaned.csv")
    save_dataframe(pd.DataFrame({"feature": feature_cols}), output_dir, "selected_features.csv")
    save_dataframe(model.artifacts.model.summary.reset_index(), output_dir, "model_coefficients.csv")
    save_dataframe(approval_policy, output_dir, "industry_approval_policy.csv")
    save_dataframe(loan_decisions, output_dir, "loan_decisions.csv")
    save_dataframe(industry_decision_summary, output_dir, "industry_decision_summary.csv")
    save_dataframe(warning_policy, output_dir, "industry_warning_policy.csv")
    save_dataframe(warning_scored, output_dir, "dynamic_warning_full.csv")
    save_dataframe(latest_warning, output_dir, "dynamic_warning_latest.csv")

    plot_top_coefficients(model.artifacts.model.summary, output_dir=output_dir, top_n=15)

    metrics = {
        "split_cutoff_date": str(pd.to_datetime(cutoff).date()),
        "n_train_loans": int(len(train_loans)),
        "n_test_loans": int(len(test_loans)),
        "train_events": int(train_loans["event"].sum()),
        "test_events": int(test_loans["event"].sum()),
        "n_train_intervals": int(len(train_intervals)),
        "n_test_intervals": int(len(test_intervals)),
        "n_features": int(len(feature_cols)),
        "train_c_index": float(train_c),
        "test_c_index": float(test_c),
        "warning_threshold_global": float(global_warning_threshold),
        "warning_precision": warn_metrics["precision"],
        "warning_recall": warn_metrics["recall"],
        "warning_f1": warn_metrics["f1"],
        "warning_avg_precision": warn_metrics["avg_precision"],
        "warning_roc_auc": warn_metrics["roc_auc"],
        "warning_positive_rate": warn_metrics["positive_rate"],
    }
    save_metrics(metrics, output_dir)

    summary_text = build_readable_summary(metrics)
    (output_dir / "experiment_summary.txt").write_text(summary_text, encoding="utf-8-sig")

    return metrics
