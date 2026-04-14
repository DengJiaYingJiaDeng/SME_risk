from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import ExperimentConfig


@dataclass
class PreparedData:
    loan_master: pd.DataFrame
    interval_data: pd.DataFrame


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.offsets.MonthEnd(0)).normalize()


def _safe_divide(a: pd.Series, b: pd.Series, eps: float = 1.0) -> pd.Series:
    return a / (b + eps)


def _standardize_sector(raw: pd.Series) -> pd.Series:
    return raw.fillna("Unknown").astype(str).str.strip()


def prepare_loan_master(loan: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    df = loan.copy()
    df = df.dropna(subset=["company_reg_number", "loan_start_date"]).copy()

    df["loan_key"] = df["loan_id"].astype(str)
    missing_key = df["loan_key"].isna() | (df["loan_key"].str.lower() == "nan")
    if missing_key.any():
        df.loc[missing_key, "loan_key"] = (
            df.loc[missing_key, "company_reg_number"].astype(str)
            + "_generated_"
            + df.loc[missing_key].index.astype(str)
        )

    snapshot = pd.Timestamp(snapshot_date)
    use_augmented_event = {"event_augmented", "event_date_augmented"}.issubset(df.columns)

    if use_augmented_event:
        df["event"] = pd.to_numeric(df["event_augmented"], errors="coerce").fillna(0).astype(int)
        df["event_date"] = pd.to_datetime(df["event_date_augmented"], errors="coerce")

        if "obs_end_date" in df.columns:
            censor_end = pd.to_datetime(df["obs_end_date"], errors="coerce")
        else:
            censor_end = df[["loan_satisfaction_date", "loan_date_due_to_close"]].min(axis=1)
        censor_end = censor_end.fillna(snapshot)
        censor_end = censor_end.where(censor_end <= snapshot, snapshot)

        valid_event = (
            (df["event"] == 1)
            & df["event_date"].notna()
            & (df["event_date"] >= df["loan_start_date"])
            & (df["event_date"] <= snapshot)
        )
        df.loc[~valid_event, "event"] = 0
        df.loc[df["event"] == 0, "event_date"] = pd.NaT
    else:
        default_mask = (df["loan_status"].astype(str).str.lower() == "defaulted") & df["loan_default_date"].notna()
        df["event_date"] = pd.NaT
        df.loc[default_mask, "event_date"] = df.loc[default_mask, "loan_default_date"]
        df["event"] = (
            default_mask
            & (df["event_date"] >= df["loan_start_date"])
            & (df["event_date"] <= snapshot)
        ).astype(int)

        censor_end = df[["loan_satisfaction_date", "loan_date_due_to_close"]].min(axis=1)
        censor_end = censor_end.fillna(snapshot)
        censor_end = censor_end.where(censor_end <= snapshot, snapshot)

    event_or_censor_end = np.where(df["event"] == 1, df["event_date"], censor_end)
    df["obs_end_date"] = pd.to_datetime(event_or_censor_end, errors="coerce")
    df["obs_end_date"] = df["obs_end_date"].fillna(snapshot)

    invalid_end = df["obs_end_date"] < df["loan_start_date"]
    df.loc[invalid_end, "obs_end_date"] = df.loc[invalid_end, "loan_start_date"] + pd.Timedelta(days=1)

    event_time_days = (df["event_date"] - df["loan_start_date"]).dt.days + 1
    df["event_time"] = np.where(df["event"] == 1, event_time_days, np.inf)
    if "duration_augmented_days" in df.columns and use_augmented_event:
        duration = pd.to_numeric(df["duration_augmented_days"], errors="coerce")
    else:
        duration = (df["obs_end_date"] - df["loan_start_date"]).dt.days + 1
    df["duration_days"] = pd.to_numeric(duration, errors="coerce").fillna(1).clip(lower=1).astype(int)

    df["industry_group"] = _standardize_sector(df["primary_sector"])
    df["loan_original_amount"] = pd.to_numeric(df["loan_original_amount"], errors="coerce").fillna(0.0)
    def _numeric_series(col: str, fallback_col: str | None = None, default: float = 0.0) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        if fallback_col is not None and fallback_col in df.columns:
            return pd.to_numeric(df[fallback_col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    df["interest"] = _numeric_series("interest")
    df["loan_number_of_missed_payments"] = _numeric_series("loan_number_of_missed_payments")
    df["total_time_payments_late"] = _numeric_series("total_time_payments_late")
    df["behavior_distress_flag"] = _numeric_series("behavior_distress_flag")
    df["origination_credit_distress_flag"] = _numeric_series("origination_credit_distress_flag")
    df["is_overdraft"] = _numeric_series("is_overdraft", fallback_col="overdraft")
    df["is_term_loan"] = _numeric_series("is_term_loan", fallback_col="loan")

    keep_cols = [
        "loan_key",
        "company_reg_number",
        "industry_group",
        "loan_start_date",
        "obs_end_date",
        "event_date",
        "event",
        "event_time",
        "duration_days",
        "loan_original_amount",
        "interest",
        "loan_number_of_missed_payments",
        "total_time_payments_late",
        "behavior_distress_flag",
        "origination_credit_distress_flag",
        "is_overdraft",
        "is_term_loan",
        "loan_status",
        "loan_repayment_frequency",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep_cols].copy()


def _expand_intervals(loan_master: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for row in loan_master.itertuples(index=False):
        cur = row.loan_start_date.normalize()
        end = row.obs_end_date.normalize()
        if end < cur:
            end = cur

        while cur <= end:
            interval_end = min(_month_end(cur), end)
            start_t = (cur - row.loan_start_date.normalize()).days
            stop_t = (interval_end - row.loan_start_date.normalize()).days + 1
            if stop_t <= start_t:
                stop_t = start_t + 1

            event_here = 0
            if row.event == 1 and pd.notna(row.event_date):
                event_ts = row.event_date.normalize()
                if cur <= event_ts <= interval_end:
                    event_here = 1

            rows.append(
                {
                    "loan_key": row.loan_key,
                    "company_reg_number": row.company_reg_number,
                    "industry_group": row.industry_group,
                    "loan_start_date": row.loan_start_date.normalize(),
                    "obs_end_date": row.obs_end_date.normalize(),
                    "event_time": row.event_time,
                    "event": event_here,
                    "start": float(start_t),
                    "stop": float(stop_t),
                    "month_end": _month_end(interval_end),
                }
            )

            if interval_end >= end:
                break
            cur = interval_end + pd.Timedelta(days=1)

    return pd.DataFrame(rows)


def _build_static_company_features(businesses: pd.DataFrame, factoring: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
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
    ]
    existing_cols = [c for c in base_cols if c in businesses.columns]
    b = businesses[existing_cols].drop_duplicates(subset=["company_reg_number"]).copy()

    numeric_defaults = [
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
    ]
    for col in numeric_defaults:
        if col in b.columns:
            b[col] = pd.to_numeric(b[col], errors="coerce")

    b["liquidity_ratio"] = _safe_divide(b["current_assets"], b["current_liabilities"])
    b["leverage_ratio"] = _safe_divide(
        b["current_liabilities"] + b["long_term_liabilities"],
        b["capital_and_reserves"],
    )
    b["profit_margin_proxy"] = _safe_divide(b["2019_revenue"] - b["costs"], b["2019_revenue"])
    b["working_capital"] = b["current_assets"] - b["current_liabilities"]
    b["receivable_turnover_proxy"] = _safe_divide(b["2019_revenue"], b["accounts_receivable"])

    f = factoring[["company_reg_number", "factor_amount", "factor_percent"]].copy()
    f["factor_amount"] = pd.to_numeric(f["factor_amount"], errors="coerce")
    f["factor_percent"] = pd.to_numeric(f["factor_percent"], errors="coerce")
    f = f.drop_duplicates(subset=["company_reg_number"])

    merged = b.merge(f, on="company_reg_number", how="left")
    return merged


def _build_macro_monthly(macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.copy()
    m = m.dropna(subset=["calendar_month"]).copy()
    m["month_end"] = m["calendar_month"].dt.to_period("M").dt.to_timestamp("M")
    m = m.sort_values("month_end").drop_duplicates(subset=["month_end"], keep="last")
    m["uk_interest_rate_diff"] = m["uk_interest_rate"].diff().fillna(0.0)
    m["uk_cpi_diff"] = m["uk_cpi"].diff().fillna(0.0)
    m["uk_gdp_yoy"] = m["uk_gdp"].pct_change(12).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return m[
        [
            "month_end",
            "uk_interest_rate",
            "uk_cpi",
            "uk_gdp",
            "uk_interest_rate_diff",
            "uk_cpi_diff",
            "uk_gdp_yoy",
        ]
    ].copy()


def _build_credit_rating_history(credit_rating: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "company_reg_number",
        "credit_report_date",
        "credit_report_total_indebtedness",
        "liens_filed_and_open",
        "liens_closed_last_five_years",
        "missed_and_late_payments_last_five_years",
        "filed_court_cases_last_five_years",
        "payment_index",
        "credit_report_negative_items",
        "credit_report_good_standing_items",
        "business_failure_score",
        "credit_report_credit_score",
        "ratio_debt_to_revenue",
    ]
    c = credit_rating[cols].dropna(subset=["company_reg_number", "credit_report_date"]).copy()
    c = c.sort_values(["company_reg_number", "credit_report_date"])
    c["credit_signal_score"] = (
        c["credit_report_good_standing_items"].fillna(0) - c["credit_report_negative_items"].fillna(0)
    )
    return c


def _build_account_history_monthly(credit_account: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "company_reg_number",
        "ca_start_date",
        "amount",
        "pay_in_amount",
        "pay_out_amount",
        "rev_ratio",
        "cost_ratio",
    ]
    c = credit_account[cols].dropna(subset=["company_reg_number", "ca_start_date"]).copy()
    c["month_end"] = c["ca_start_date"].dt.to_period("M").dt.to_timestamp("M")

    g = (
        c.groupby(["company_reg_number", "month_end"])
        .agg(
            ca_monthly_new_accounts=("ca_start_date", "count"),
            ca_monthly_amount=("amount", "sum"),
            ca_monthly_payin=("pay_in_amount", "sum"),
            ca_monthly_payout=("pay_out_amount", "sum"),
            ca_latest_rev_ratio=("rev_ratio", "mean"),
            ca_latest_cost_ratio=("cost_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["company_reg_number", "month_end"])
    )

    for src, dst in [
        ("ca_monthly_new_accounts", "ca_cum_new_accounts"),
        ("ca_monthly_amount", "ca_cum_amount"),
        ("ca_monthly_payin", "ca_cum_payin"),
        ("ca_monthly_payout", "ca_cum_payout"),
    ]:
        g[dst] = g.groupby("company_reg_number")[src].cumsum()

    g["ca_latest_rev_ratio"] = g.groupby("company_reg_number")["ca_latest_rev_ratio"].ffill()
    g["ca_latest_cost_ratio"] = g.groupby("company_reg_number")["ca_latest_cost_ratio"].ffill()

    keep = [
        "company_reg_number",
        "month_end",
        "ca_cum_new_accounts",
        "ca_cum_amount",
        "ca_cum_payin",
        "ca_cum_payout",
        "ca_latest_rev_ratio",
        "ca_latest_cost_ratio",
    ]
    return g[keep].copy()


def _build_card_history_monthly(credit_card: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "company_reg_number",
        "cc_start_date",
        "cc_agreed_limit",
        "cc_balance_limit_ratio",
        "missed_payments_number",
    ]
    c = credit_card[cols].dropna(subset=["company_reg_number", "cc_start_date"]).copy()
    c["month_end"] = c["cc_start_date"].dt.to_period("M").dt.to_timestamp("M")

    g = (
        c.groupby(["company_reg_number", "month_end"])
        .agg(
            cc_monthly_new_cards=("cc_start_date", "count"),
            cc_monthly_agreed_limit=("cc_agreed_limit", "sum"),
            cc_latest_balance_limit_ratio=("cc_balance_limit_ratio", "mean"),
            cc_monthly_missed_payments=("missed_payments_number", "sum"),
        )
        .reset_index()
        .sort_values(["company_reg_number", "month_end"])
    )

    g["cc_cum_new_cards"] = g.groupby("company_reg_number")["cc_monthly_new_cards"].cumsum()
    g["cc_cum_agreed_limit"] = g.groupby("company_reg_number")["cc_monthly_agreed_limit"].cumsum()
    g["cc_cum_missed_payments"] = g.groupby("company_reg_number")["cc_monthly_missed_payments"].cumsum()
    g["cc_latest_balance_limit_ratio"] = g.groupby("company_reg_number")["cc_latest_balance_limit_ratio"].ffill()

    keep = [
        "company_reg_number",
        "month_end",
        "cc_cum_new_cards",
        "cc_cum_agreed_limit",
        "cc_latest_balance_limit_ratio",
        "cc_cum_missed_payments",
    ]
    return g[keep].copy()


def _merge_asof_by_company(left: pd.DataFrame, right: pd.DataFrame, right_time_col: str = "month_end") -> pd.DataFrame:
    if right.empty:
        return left

    l = left.copy()
    r = right.copy()
    l["month_end"] = pd.to_datetime(l["month_end"], errors="coerce")
    r[right_time_col] = pd.to_datetime(r[right_time_col], errors="coerce")
    l = l.dropna(subset=["company_reg_number", "month_end"])
    r = r.dropna(subset=["company_reg_number", right_time_col])
    l["company_reg_number"] = pd.to_numeric(l["company_reg_number"], errors="coerce").astype("int64")
    r["company_reg_number"] = pd.to_numeric(r["company_reg_number"], errors="coerce").astype("int64")

    payload_cols = [c for c in r.columns if c not in {"company_reg_number", right_time_col}]
    merged_groups: List[pd.DataFrame] = []
    for company_id, l_grp in l.groupby("company_reg_number", sort=False):
        l_grp = l_grp.sort_values("month_end").copy()
        r_grp = r[r["company_reg_number"] == company_id].sort_values(right_time_col).copy()

        if r_grp.empty:
            for col in payload_cols:
                l_grp[col] = np.nan
            merged_groups.append(l_grp)
            continue

        right_payload = r_grp[[right_time_col] + payload_cols]
        if right_time_col == "month_end":
            merged = pd.merge_asof(
                l_grp,
                right_payload,
                on="month_end",
                direction="backward",
                suffixes=("", "_asof"),
            )
        else:
            merged = pd.merge_asof(
                l_grp,
                right_payload,
                left_on="month_end",
                right_on=right_time_col,
                direction="backward",
                suffixes=("", "_asof"),
            )
            if right_time_col in merged.columns:
                merged = merged.drop(columns=[right_time_col])
        merged_groups.append(merged)

    return pd.concat(merged_groups, ignore_index=True)


def _add_log_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            safe = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
            df[f"log_{col}"] = np.log1p(safe)
    return df


def build_modeling_dataset(tables: Dict[str, pd.DataFrame], config: ExperimentConfig) -> PreparedData:
    loan_master = prepare_loan_master(tables["loan"], snapshot_date=config.snapshot_date)
    intervals = _expand_intervals(loan_master)

    static_company = _build_static_company_features(tables["businesses"], tables["factoring"])
    macro = _build_macro_monthly(tables["macro"])
    rating = _build_credit_rating_history(tables["credit_rating"])
    account_monthly = _build_account_history_monthly(tables["credit_account"])
    card_monthly = _build_card_history_monthly(tables["credit_card"])

    intervals = intervals.merge(
        loan_master[
            [
                "loan_key",
                "loan_original_amount",
                "interest",
                "loan_number_of_missed_payments",
                "total_time_payments_late",
                "behavior_distress_flag",
                "origination_credit_distress_flag",
                "is_overdraft",
                "is_term_loan",
                "duration_days",
            ]
        ],
        on="loan_key",
        how="left",
    )
    intervals = intervals.merge(static_company, on="company_reg_number", how="left")
    intervals = intervals.merge(macro, on="month_end", how="left")

    intervals = _merge_asof_by_company(
        intervals,
        rating.rename(columns={"credit_report_date": "month_end"}),
        right_time_col="month_end",
    )
    intervals = _merge_asof_by_company(intervals, account_monthly, right_time_col="month_end")
    intervals = _merge_asof_by_company(intervals, card_monthly, right_time_col="month_end")

    intervals["loan_age_days"] = intervals["stop"]
    intervals["log_loan_amount"] = np.log1p(intervals["loan_original_amount"].clip(lower=0))
    intervals = _add_log_features(
        intervals,
        [
            "factor_amount",
            "ca_cum_amount",
            "ca_cum_payin",
            "ca_cum_payout",
            "cc_cum_agreed_limit",
        ],
    )

    intervals["industry_group"] = _standardize_sector(intervals["industry_group"])

    # Merge loan-level event flag for horizon labels.
    intervals = intervals.merge(
        loan_master[["loan_key", "event"]],
        on="loan_key",
        how="left",
        suffixes=("", "_loan"),
    )
    intervals = intervals.rename(columns={"event_loan": "loan_event"})
    intervals["loan_event"] = pd.to_numeric(intervals.get("loan_event"), errors="coerce").fillna(0).astype(int)
    intervals["loan_event_time"] = pd.to_numeric(intervals.get("event_time"), errors="coerce")

    # Fill global macro missing with forward/backward fill.
    macro_cols = [
        "uk_interest_rate",
        "uk_cpi",
        "uk_gdp",
        "uk_interest_rate_diff",
        "uk_cpi_diff",
        "uk_gdp_yoy",
    ]
    for col in macro_cols:
        if col in intervals.columns:
            intervals[col] = intervals[col].ffill().bfill()

    intervals = intervals.sort_values(["loan_key", "start"]).reset_index(drop=True)
    return PreparedData(loan_master=loan_master, interval_data=intervals)


def select_feature_columns(interval_data: pd.DataFrame, config: ExperimentConfig) -> List[str]:
    selected: List[str] = []
    for col in config.preferred_feature_order:
        if col in interval_data.columns and pd.api.types.is_numeric_dtype(interval_data[col]):
            if interval_data[col].nunique(dropna=True) > 1:
                selected.append(col)

    if not selected:
        protected = {
            "start",
            "stop",
            "event",
            "loan_event",
            "loan_event_time",
            "duration_days",
            "company_reg_number",
        }
        for col in interval_data.columns:
            if col in protected or col.endswith("_date"):
                continue
            if col in {"loan_key", "industry_group", "month_end"}:
                continue
            if pd.api.types.is_numeric_dtype(interval_data[col]) and interval_data[col].nunique(dropna=True) > 1:
                selected.append(col)
    return selected
