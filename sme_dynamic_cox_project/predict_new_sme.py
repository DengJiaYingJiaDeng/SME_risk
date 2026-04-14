from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.sme_dynamic_cox.config import ExperimentConfig
from src.sme_dynamic_cox.data_io import load_all_tables
from src.sme_dynamic_cox.evaluation import add_future_event_label, build_origination_view
from src.sme_dynamic_cox.feature_builder import build_modeling_dataset, select_feature_columns
from src.sme_dynamic_cox.model import DynamicCoxRiskModel
from src.sme_dynamic_cox.policy import (
    apply_approval_policy,
    apply_warning_policy,
    build_industry_approval_policy,
    build_industry_warning_policy,
)


def _read_csv_fallback(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"Cannot read file: {path}")


def _to_datetime_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", format="mixed")
    return out


def _to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _append_if_exists(base_df: pd.DataFrame, new_path: str | None) -> pd.DataFrame:
    if not new_path:
        return base_df
    p = Path(new_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    new_df = _read_csv_fallback(p)
    return pd.concat([base_df, new_df], axis=0, ignore_index=True, sort=False)


def _ensure_new_loan_ids(new_loan: pd.DataFrame, historical_loan_ids: set[str]) -> pd.DataFrame:
    df = new_loan.copy()
    if "loan_id" not in df.columns:
        df["loan_id"] = ""
    df["loan_id"] = df["loan_id"].astype(str).str.strip()

    used = set(historical_loan_ids)
    for i in df.index:
        cur = str(df.at[i, "loan_id"])
        if (cur == "") or (cur.lower() == "nan") or (cur in used):
            new_id = f"NEW_LOAN_{i}"
            while new_id in used:
                new_id = f"{new_id}_X"
            df.at[i, "loan_id"] = new_id
            used.add(new_id)
        else:
            used.add(cur)
    return df


def _top_risk_factors(model: DynamicCoxRiskModel, df: pd.DataFrame, top_n: int = 3) -> pd.Series:
    contrib = model.local_feature_contributions(df)
    labels = []
    for idx in df.index:
        c = contrib.loc[idx].sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)
        factor_text = "; ".join([f"{k}({v:+.3f})" for k, v in c.items()])
        labels.append(factor_text)
    return pd.Series(labels, index=df.index, name="top_risk_factors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict risk and decision for new SME loan applications.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"E:\my model\SME_model\SME - Synthetic UK Businesses financial statistics",
        help="Historical dataset directory.",
    )
    parser.add_argument(
        "--base-loan-file",
        type=str,
        default="Loan_Augmented_RealSignals.csv",
        help="Historical loan file under data-dir.",
    )
    parser.add_argument("--new-loan-file", type=str, required=True, help="CSV path for new loan applications.")
    parser.add_argument("--new-business-file", type=str, default="", help="Optional new business records CSV.")
    parser.add_argument("--new-credit-rating-file", type=str, default="", help="Optional new credit rating CSV.")
    parser.add_argument("--new-credit-account-file", type=str, default="", help="Optional new credit account CSV.")
    parser.add_argument("--new-factoring-file", type=str, default="", help="Optional new factoring CSV.")
    parser.add_argument("--new-credit-card-file", type=str, default="", help="Optional new credit card CSV.")
    parser.add_argument("--snapshot-date", type=str, default="2024-02-29", help="Snapshot date.")
    parser.add_argument("--warning-horizon-days", type=int, default=90, help="Warning horizon.")
    parser.add_argument(
        "--output-file",
        type=str,
        default=r"E:\my model\SME_model\sme_dynamic_cox_project\outputs\new_sme_predictions.csv",
        help="Prediction output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tables = load_all_tables(data_dir, loan_file_name=args.base_loan_file)
    historical_loan = tables["loan"].copy()
    historical_loan_ids = set(historical_loan.get("loan_id", pd.Series(dtype=str)).astype(str).tolist())

    new_loan = _read_csv_fallback(Path(args.new_loan_file))
    new_loan = _to_datetime_cols(
        new_loan,
        [
            "loan_start_date",
            "loan_date_due_to_close",
            "loan_default_date",
            "loan_satisfaction_date",
            "obs_end_date",
            "event_date_augmented",
        ],
    )
    new_loan = _to_numeric_cols(
        new_loan,
        [
            "company_reg_number",
            "loan_original_amount",
            "interest",
            "loan_number_of_missed_payments",
            "loan",
            "overdraft",
            "is_overdraft",
            "is_term_loan",
            "event_augmented",
            "duration_augmented_days",
            "total_time_payments_late",
        ],
    )
    new_loan = _ensure_new_loan_ids(new_loan, historical_loan_ids)

    tables["loan"] = pd.concat([historical_loan, new_loan], axis=0, ignore_index=True, sort=False)
    tables["businesses"] = _append_if_exists(tables["businesses"], args.new_business_file)
    tables["businesses"] = _to_numeric_cols(
        tables["businesses"],
        [
            "company_reg_number",
            "women_owned",
            "number_of_officers",
            "2019_revenue",
            "capex",
            "cogs",
            "costs",
            "accounts_receivable",
            "capital_and_reserves",
            "current_assets",
            "current_liabilities",
            "fixed_assets",
            "long_term_liabilities",
            "provisions_for_liabilities",
        ],
    )

    tables["credit_rating"] = _append_if_exists(tables["credit_rating"], args.new_credit_rating_file)
    tables["credit_rating"] = _to_datetime_cols(tables["credit_rating"], ["credit_report_date"])
    tables["credit_rating"] = _to_numeric_cols(
        tables["credit_rating"],
        [
            "company_reg_number",
            "credit_report_total_indebtedness",
            "payment_index",
            "credit_report_negative_items",
            "credit_report_good_standing_items",
            "business_failure_score",
            "credit_report_credit_score",
            "ratio_debt_to_revenue",
            "missed_and_late_payments_last_five_years",
        ],
    )

    tables["credit_account"] = _append_if_exists(tables["credit_account"], args.new_credit_account_file)
    tables["credit_account"] = _to_datetime_cols(tables["credit_account"], ["ca_start_date"])
    tables["credit_account"] = _to_numeric_cols(
        tables["credit_account"],
        [
            "company_reg_number",
            "amount",
            "pay_in_amount",
            "pay_out_amount",
            "rev_ratio",
            "cost_ratio",
            "number_of_accounts",
            "current_acc_fraction",
        ],
    )

    tables["factoring"] = _append_if_exists(tables["factoring"], args.new_factoring_file)
    tables["factoring"] = _to_numeric_cols(tables["factoring"], ["company_reg_number", "factor_amount", "factor_percent"])

    tables["credit_card"] = _append_if_exists(tables["credit_card"], args.new_credit_card_file)
    tables["credit_card"] = _to_datetime_cols(tables["credit_card"], ["cc_start_date"])
    tables["credit_card"] = _to_numeric_cols(
        tables["credit_card"],
        ["company_reg_number", "cc_agreed_limit", "cc_balance_limit_ratio", "missed_payments_number"],
    )

    cfg = ExperimentConfig(
        data_dir=data_dir,
        output_dir=output_file.parent,
        loan_file_name=args.base_loan_file,
        snapshot_date=args.snapshot_date,
        warning_horizon_days=args.warning_horizon_days,
    )

    prepared = build_modeling_dataset(tables, cfg)
    loan_master = prepared.loan_master.copy()
    interval_data = prepared.interval_data.copy()
    feature_cols = select_feature_columns(interval_data, cfg)

    new_keys = set(new_loan["loan_id"].astype(str))
    train_intervals = interval_data[~interval_data["loan_key"].astype(str).isin(new_keys)].copy()
    if train_intervals["event"].sum() < 3:
        raise RuntimeError("Historical events are insufficient for stable model training.")

    model = DynamicCoxRiskModel(penalizer=cfg.penalizer, l1_ratio=cfg.l1_ratio, strata_col="industry_group")
    model.fit(train_intervals, feature_cols)

    origination_df = build_origination_view(interval_data, loan_master)
    origination_df["risk_score"] = model.predict_partial_hazard(origination_df)
    origination_df["p_default_90d"] = model.predict_default_probability(origination_df, horizon_days=90)
    origination_df["p_default_180d"] = model.predict_default_probability(origination_df, horizon_days=180)

    historical_orig = origination_df[~origination_df["loan_key"].astype(str).isin(new_keys)].copy()
    new_orig = origination_df[origination_df["loan_key"].astype(str).isin(new_keys)].copy()

    approval_policy = build_industry_approval_policy(
        historical_orig,
        industry_col="industry_group",
        score_col="risk_score",
        target_col="event",
        min_samples_per_industry=cfg.min_samples_per_industry,
        gain=cfg.loan_approval_gain,
        loss=cfg.loan_default_loss,
    )
    decision_df = apply_approval_policy(
        pd.concat([historical_orig, new_orig], axis=0, ignore_index=True),
        policy_df=approval_policy,
        industry_col="industry_group",
        score_col="risk_score",
    )
    new_decisions = decision_df[decision_df["loan_key"].astype(str).isin(new_keys)].copy()

    monitored_df = add_future_event_label(interval_data, horizon_days=cfg.warning_horizon_days)
    monitored_df = monitored_df[monitored_df["monitoring_row"] == 1].copy()
    prob_col = f"p_default_{cfg.warning_horizon_days}d"
    monitored_df[prob_col] = model.predict_default_probability(monitored_df, horizon_days=cfg.warning_horizon_days)

    historical_monitor = monitored_df[~monitored_df["loan_key"].astype(str).isin(new_keys)].copy()
    warning_policy = build_industry_warning_policy(
        historical_monitor,
        industry_col="industry_group",
        prob_col=prob_col,
        target_col="future_event_label",
        min_samples_per_industry=30,
    )
    # For new applications, use origination 90-day probability with learned warning thresholds.
    new_warning = apply_warning_policy(
        new_decisions.copy(),
        policy_df=warning_policy,
        industry_col="industry_group",
        prob_col="p_default_90d",
    )

    new_orig = new_orig.copy()
    new_orig["top_risk_factors"] = _top_risk_factors(model, new_orig, top_n=3)

    out = new_decisions[
        [
            "loan_key",
            "company_reg_number",
            "industry_group",
            "loan_start_date",
            "risk_score",
            "p_default_90d",
            "p_default_180d",
            "loan_decision",
            "threshold",
            "threshold_source",
        ]
    ].copy()
    out = out.merge(
        new_warning[["loan_key", "risk_level", "warning_flag", "warning_threshold", "warning_threshold_source"]],
        on="loan_key",
        how="left",
    )
    out = out.merge(new_orig[["loan_key", "top_risk_factors"]], on="loan_key", how="left")
    out.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("=== New SME Prediction Completed ===")
    print(f"Scored loans: {len(out)}")
    print(f"Output: {output_file}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
