from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data_io import load_all_tables
from .evaluation import (
    add_future_event_label,
    brier_score,
    brier_skill_score,
    build_origination_view,
    build_tradeoff_curve,
    choose_time_split,
    classification_metrics,
    evaluate_time_dependent_metrics,
    ks_statistic,
)
from .feature_builder import build_modeling_dataset, select_feature_columns
from .model import DynamicCoxRiskModel
from .policy import (
    apply_approval_policy,
    apply_warning_policy,
    build_industry_approval_policy,
    build_industry_warning_policy,
)
from .reporting import (
    build_readable_summary,
    plot_local_waterfall,
    plot_pdp_curves,
    plot_stratified_survival,
    plot_time_dependent_curves,
    plot_top_coefficients,
    plot_tradeoff_curve,
    save_dataframe,
    save_metrics,
)


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


def _build_stratified_summary(test_orig: pd.DataFrame, split_threshold: float) -> pd.DataFrame:
    df = test_orig.copy()
    df["risk_group"] = np.where(df["risk_score"] >= split_threshold, "High-Risk", "Prime")
    out = (
        df.groupby("risk_group")
        .agg(
            n_loans=("loan_key", "count"),
            event_rate=("event", "mean"),
            median_duration_days=("duration_days", "median"),
            avg_risk_score=("risk_score", "mean"),
        )
        .reset_index()
    )
    out["split_threshold"] = split_threshold
    return out


def _build_local_explanations(
    model: DynamicCoxRiskModel,
    sample_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()

    contrib = model.local_feature_contributions(sample_df)
    z = model.transform_features(sample_df)
    rows: List[Dict] = []

    for idx, row in sample_df.iterrows():
        loan_key = str(row["loan_key"])
        risk_score = float(row["risk_score"])
        c = contrib.loc[idx].sort_values(key=lambda x: x.abs(), ascending=False)
        z_row = z.loc[idx]
        one = pd.DataFrame(
            {
                "loan_key": loan_key,
                "feature": c.index,
                "contribution": c.values,
                "standardized_value": z_row.reindex(c.index).values,
                "risk_score": risk_score,
            }
        )
        rows.append(one)

        safe_key = (
            loan_key.replace("\\", "_")
            .replace("/", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )
        plot_local_waterfall(
            contribution_series=c,
            output_path=output_dir / f"local_explain_waterfall_{safe_key}.png",
            title=f"Local Explanation Waterfall: {loan_key}",
            top_n=top_n,
        )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_pdp_frame(
    model: DynamicCoxRiskModel,
    base_df: pd.DataFrame,
    feature_candidates: List[str],
    points: int = 20,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for feat in feature_candidates:
        if feat not in base_df.columns:
            continue
        x = pd.to_numeric(base_df[feat], errors="coerce").dropna()
        if x.nunique() < 5:
            continue
        grid = np.unique(np.quantile(x, np.linspace(0.05, 0.95, points)))
        for val in grid:
            temp = base_df.copy()
            temp[feat] = float(val)
            mean_h = float(model.predict_partial_hazard(temp).mean())
            rows.append({"feature": feat, "feature_value": float(val), "mean_partial_hazard": mean_h})
    return pd.DataFrame(rows)


def run_experiment(config: ExperimentConfig) -> Dict:
    tables = load_all_tables(config.data_dir, loan_file_name=config.loan_file_name)
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
    eval_horizons = sorted(set(int(h) for h in config.evaluation_horizons_days if int(h) > 0))
    for h in eval_horizons:
        origination_df[f"p_default_{h}d_orig"] = model.predict_default_probability(origination_df, horizon_days=h)

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
    prob_col = f"p_default_{config.warning_horizon_days}d"
    monitored_df[prob_col] = model.predict_default_probability(monitored_df, horizon_days=config.warning_horizon_days)

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
    warning_ks_90d = ks_statistic(test_warning["future_event_label"], test_warning[prob_col])
    warning_brier_90d = brier_score(test_warning["future_event_label"], test_warning[prob_col])
    warning_brier_skill_90d = brier_skill_score(test_warning["future_event_label"], test_warning[prob_col])

    latest_warning = (
        warning_scored.sort_values(["loan_key", "month_end", prob_col]).groupby("loan_key", as_index=False).tail(1)
    )
    latest_warning = latest_warning.sort_values(["risk_level", prob_col], ascending=[True, False]).reset_index(drop=True)

    # Time-dependent AUC / cumulative AUC / Brier / KS on test origination cohort.
    prob_by_horizon = {h: test_orig[f"p_default_{h}d_orig"] for h in eval_horizons if f"p_default_{h}d_orig" in test_orig.columns}
    time_dep_metrics = evaluate_time_dependent_metrics(
        durations=test_orig["duration_days"],
        events=test_orig["event"],
        prob_by_horizon=prob_by_horizon,
    )
    integrated_brier = float(time_dep_metrics["brier_score"].mean()) if not time_dep_metrics.empty else float("nan")

    # Stratified survival plot data.
    split_threshold = float(train_orig["risk_score"].median()) if not train_orig.empty else float(origination_df["risk_score"].median())
    stratified_summary = _build_stratified_summary(test_orig, split_threshold=split_threshold)

    # Trade-off curve on test cohort.
    tradeoff_curve = build_tradeoff_curve(test_orig["risk_score"], test_orig["event"])
    bad_rate_target = 0.03
    feasible = tradeoff_curve[tradeoff_curve["bad_debt_rate"] <= bad_rate_target]
    max_approval_at_3pct = float(feasible["approval_rate"].max()) if not feasible.empty else 0.0

    # Local explainability samples (highest and lowest risk on test set).
    explain_samples = pd.concat(
        [
            test_orig.sort_values("risk_score", ascending=False).head(1),
            test_orig.sort_values("risk_score", ascending=True).head(1),
        ],
        axis=0,
    ).drop_duplicates(subset=["loan_key"])
    local_explanations = _build_local_explanations(model, explain_samples, output_dir=Path(config.output_dir))

    # PDP for two strongest features.
    coef_abs = model.artifacts.model.summary["coef"].abs().sort_values(ascending=False)
    pdp_features = [f for f in coef_abs.index.tolist() if f in feature_cols][:2]
    pdp_frame = _build_pdp_frame(model, train_orig, pdp_features, points=20)

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
    save_dataframe(time_dep_metrics, output_dir, "time_dependent_metrics.csv")
    save_dataframe(tradeoff_curve, output_dir, "strategy_tradeoff_curve.csv")
    save_dataframe(stratified_summary, output_dir, "stratified_survival_summary.csv")
    save_dataframe(local_explanations, output_dir, "local_explanations.csv")
    save_dataframe(pdp_frame, output_dir, "partial_dependence_data.csv")

    plot_top_coefficients(model.artifacts.model.summary, output_dir=output_dir, top_n=15)
    plot_time_dependent_curves(time_dep_metrics, output_dir=output_dir)
    plot_stratified_survival(
        test_orig,
        score_col="risk_score",
        duration_col="duration_days",
        event_col="event",
        split_threshold=split_threshold,
        output_dir=output_dir,
    )
    plot_tradeoff_curve(tradeoff_curve, output_dir=output_dir, bad_rate_target=bad_rate_target)
    plot_pdp_curves(pdp_frame, output_dir=output_dir, feature_order=pdp_features)

    metrics = {
        "loan_file_name": config.loan_file_name,
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
        "warning_ks_90d": float(warning_ks_90d),
        "warning_brier_90d": float(warning_brier_90d),
        "warning_brier_skill_90d": float(warning_brier_skill_90d),
        "integrated_brier_score_origination": float(integrated_brier),
        "max_approval_rate_at_3pct_bad_debt": float(max_approval_at_3pct),
        "risk_split_threshold": float(split_threshold),
    }

    for row in time_dep_metrics.itertuples(index=False):
        h = int(row.horizon_days)
        metrics[f"time_auc_{h}d"] = float(row.time_dependent_auc)
        metrics[f"cumulative_auc_{h}d"] = float(row.cumulative_auc)
        metrics[f"ks_{h}d"] = float(row.ks_statistic)
        metrics[f"brier_{h}d"] = float(row.brier_score)
        metrics[f"brier_skill_{h}d"] = float(row.brier_skill_score)

    save_metrics(metrics, output_dir)
    summary_text = build_readable_summary(metrics)
    (output_dir / "experiment_summary.txt").write_text(summary_text, encoding="utf-8-sig")
    return metrics
