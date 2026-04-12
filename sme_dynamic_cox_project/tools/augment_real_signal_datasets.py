from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class AugmentConfig:
    data_dir: Path
    snapshot_date: pd.Timestamp = pd.Timestamp("2024-02-29")
    output_loan_augmented: str = "Loan_Augmented_RealSignals.csv"
    output_timevarying_augmented: str = "Loan_Augmented_TimeVarying.csv"
    output_auxiliary: str = "SME_Auxiliary_Risk_Pretrain.csv"
    output_report: str = "Data_Augmentation_Report.txt"


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"Cannot read csv: {path}")


def _to_dt(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
            miss = parsed.isna() & df[c].notna()
            if miss.any():
                parsed2 = pd.to_datetime(df.loc[miss, c], errors="coerce", format="mixed", dayfirst=True)
                parsed.loc[miss] = parsed2
            df[c] = parsed
    return df


def _to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.offsets.MonthEnd(0)).normalize()


def _safe_ratio(a: pd.Series, b: pd.Series, eps: float = 1.0) -> pd.Series:
    return a / (b + eps)


def load_tables(data_dir: Path) -> Dict[str, pd.DataFrame]:
    tables = {
        "loan": _read_csv(data_dir / "Loan.csv"),
        "businesses": _read_csv(data_dir / "Businesses.csv"),
        "credit_rating": _read_csv(data_dir / "Credit_Rating.csv"),
        "credit_account": _read_csv(data_dir / "Credit_Account_History.csv"),
        "macro": _read_csv(data_dir / "Combined_macro_data.csv"),
        "factoring": _read_csv(data_dir / "Factoring.csv"),
    }

    _to_dt(
        tables["loan"],
        ["loan_start_date", "loan_date_due_to_close", "loan_default_date", "loan_satisfaction_date"],
    )
    _to_num(
        tables["loan"],
        [
            "company_reg_number",
            "loan_original_amount",
            "interest",
            "loan_number_of_missed_payments",
            "loan_amount_outstanding_including_future_interest",
            "total_time_payments_late",
            "loan",
            "overdraft",
        ],
    )

    _to_dt(tables["businesses"], ["incorporation_date", "filing_date"])
    _to_num(
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

    _to_dt(tables["credit_rating"], ["credit_report_date"])
    _to_num(
        tables["credit_rating"],
        [
            "company_reg_number",
            "missed_and_late_payments_last_five_years",
            "business_failure_score",
            "credit_report_credit_score",
            "credit_report_negative_items",
            "payment_index",
            "ratio_debt_to_revenue",
            "credit_report_total_indebtedness",
            "credit_report_good_standing_items",
        ],
    )

    _to_dt(tables["credit_account"], ["ca_start_date"])
    _to_num(
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

    _to_dt(tables["macro"], ["calendar_month"])
    _to_num(tables["macro"], ["uk_interest_rate", "uk_cpi", "uk_gdp"])

    _to_num(tables["factoring"], ["company_reg_number", "factor_amount", "factor_percent"])
    return tables


def _latest_credit_snapshot(credit_rating: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    c = credit_rating.dropna(subset=["company_reg_number", "credit_report_date"]).copy()
    c = c[c["credit_report_date"] <= snapshot_date].copy()
    c = c.sort_values(["company_reg_number", "credit_report_date"]).groupby("company_reg_number", as_index=False).tail(1)
    keep = [
        "company_reg_number",
        "credit_report_date",
        "missed_and_late_payments_last_five_years",
        "business_failure_score",
        "credit_report_credit_score",
        "credit_report_negative_items",
        "payment_index",
        "ratio_debt_to_revenue",
        "credit_report_total_indebtedness",
        "credit_report_good_standing_items",
    ]
    return c[keep].copy()


def build_loan_augmented(tables: Dict[str, pd.DataFrame], cfg: AugmentConfig) -> pd.DataFrame:
    loan = tables["loan"].copy()
    loan = loan.dropna(subset=["company_reg_number", "loan_start_date"]).copy()
    loan["loan_key"] = loan["loan_id"].astype(str)
    missing_key = loan["loan_key"].isna() | (loan["loan_key"].str.lower() == "nan")
    if missing_key.any():
        loan.loc[missing_key, "loan_key"] = (
            loan.loc[missing_key, "company_reg_number"].astype("Int64").astype(str)
            + "_generated_"
            + loan.loc[missing_key].index.astype(str)
        )

    loan["loan_status_str"] = loan["loan_status"].astype(str).str.lower().str.strip()
    loan["event_hard_default"] = (
        (loan["loan_status_str"] == "defaulted")
        & loan["loan_default_date"].notna()
        & (loan["loan_default_date"] <= cfg.snapshot_date)
        & (loan["loan_default_date"] >= loan["loan_start_date"])
    ).astype(int)

    loan["obs_end_date"] = loan[["loan_satisfaction_date", "loan_date_due_to_close"]].min(axis=1)
    loan["obs_end_date"] = loan["obs_end_date"].fillna(cfg.snapshot_date)
    loan.loc[loan["obs_end_date"] > cfg.snapshot_date, "obs_end_date"] = cfg.snapshot_date
    invalid_end = loan["obs_end_date"] < loan["loan_start_date"]
    loan.loc[invalid_end, "obs_end_date"] = loan.loc[invalid_end, "loan_start_date"] + pd.Timedelta(days=1)

    credit_snapshot = _latest_credit_snapshot(tables["credit_rating"], cfg.snapshot_date)
    loan = loan.merge(credit_snapshot, on="company_reg_number", how="left")

    loan["behavior_distress_flag"] = (
        (loan["loan_number_of_missed_payments"].fillna(0) >= 2)
        | (loan["total_time_payments_late"].fillna(0) >= 30)
    ).astype(int)

    loan["origination_credit_distress_flag"] = (
        (loan["credit_report_date"].notna())
        & (loan["credit_report_date"] <= loan["loan_start_date"])
        & (
            (loan["business_failure_score"].fillna(-1) >= 75)
            | (loan["credit_report_credit_score"].fillna(1000) <= 520)
            | (loan["missed_and_late_payments_last_five_years"].fillna(0) >= 5)
            | (loan["credit_report_negative_items"].fillna(0) >= 6)
        )
    ).astype(int)

    loan["event_distress"] = (
        (loan["event_hard_default"] == 0)
        & (
            (loan["behavior_distress_flag"] == 1)
            | (loan["origination_credit_distress_flag"] == 1)
        )
    ).astype(int)

    loan["event_augmented"] = ((loan["event_hard_default"] == 1) | (loan["event_distress"] == 1)).astype(int)

    # Event date assignment:
    # - hard default: real default date
    # - behavioral distress: proxy at 60% of observed loan lifespan (at least day 30)
    # - origination distress: proxy at day 30 after origination
    loan["obs_duration_days"] = (loan["obs_end_date"] - loan["loan_start_date"]).dt.days + 1
    loan["obs_duration_days"] = loan["obs_duration_days"].clip(lower=1)

    behavior_day = np.minimum(
        np.maximum((loan["obs_duration_days"] * 0.6).round().astype(int), 30),
        np.maximum(loan["obs_duration_days"] - 1, 1),
    )
    orig_day = np.minimum(np.maximum(30, 7), np.maximum(loan["obs_duration_days"] - 1, 1))

    loan["event_date_augmented"] = pd.NaT
    hard_mask = loan["event_hard_default"] == 1
    behavior_mask = (loan["event_hard_default"] == 0) & (loan["behavior_distress_flag"] == 1)
    orig_mask = (
        (loan["event_hard_default"] == 0)
        & (loan["behavior_distress_flag"] == 0)
        & (loan["origination_credit_distress_flag"] == 1)
    )

    loan.loc[hard_mask, "event_date_augmented"] = loan.loc[hard_mask, "loan_default_date"]
    loan.loc[behavior_mask, "event_date_augmented"] = (
        loan.loc[behavior_mask, "loan_start_date"] + pd.to_timedelta(behavior_day[behavior_mask] - 1, unit="D")
    )
    loan.loc[orig_mask, "event_date_augmented"] = (
        loan.loc[orig_mask, "loan_start_date"] + pd.to_timedelta(orig_day[orig_mask] - 1, unit="D")
    )

    loan.loc[loan["event_date_augmented"] > loan["obs_end_date"], "event_date_augmented"] = loan["obs_end_date"]
    loan.loc[loan["event_date_augmented"] < loan["loan_start_date"], "event_date_augmented"] = loan["loan_start_date"]

    loan["event_type_augmented"] = np.select(
        [
            loan["event_hard_default"] == 1,
            behavior_mask,
            orig_mask,
        ],
        [
            "hard_default",
            "distress_behavior",
            "distress_origination",
        ],
        default="censored",
    )

    loan["duration_augmented_days"] = np.where(
        loan["event_augmented"] == 1,
        (loan["event_date_augmented"] - loan["loan_start_date"]).dt.days + 1,
        (loan["obs_end_date"] - loan["loan_start_date"]).dt.days + 1,
    )
    loan["duration_augmented_days"] = pd.to_numeric(loan["duration_augmented_days"], errors="coerce").fillna(1).clip(lower=1)

    loan["industry_group"] = loan["primary_sector"].fillna("Unknown").astype(str).str.strip()
    loan["is_overdraft"] = loan["overdraft"].fillna(0)
    loan["is_term_loan"] = loan["loan"].fillna(0)

    return loan


def _build_macro_monthly(macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.copy()
    m = m.dropna(subset=["calendar_month"]).copy()
    m["month_end"] = m["calendar_month"].dt.to_period("M").dt.to_timestamp("M")
    m = m.sort_values("month_end").drop_duplicates("month_end", keep="last")
    m["uk_interest_rate_diff"] = m["uk_interest_rate"].diff().fillna(0)
    m["uk_cpi_diff"] = m["uk_cpi"].diff().fillna(0)
    m["uk_gdp_yoy"] = m["uk_gdp"].pct_change(12).replace([np.inf, -np.inf], np.nan).fillna(0)
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


def _build_static_business_features(businesses: pd.DataFrame, factoring: pd.DataFrame) -> pd.DataFrame:
    cols = [
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
    b = businesses[cols].drop_duplicates("company_reg_number").copy()
    b["liquidity_ratio"] = _safe_ratio(b["current_assets"], b["current_liabilities"])
    b["leverage_ratio"] = _safe_ratio(
        b["current_liabilities"].fillna(0) + b["long_term_liabilities"].fillna(0),
        b["capital_and_reserves"],
    )
    b["profit_margin_proxy"] = _safe_ratio(b["2019_revenue"] - b["costs"], b["2019_revenue"])
    b["working_capital"] = b["current_assets"] - b["current_liabilities"]
    b["receivable_turnover_proxy"] = _safe_ratio(b["2019_revenue"], b["accounts_receivable"])

    f = factoring[["company_reg_number", "factor_amount", "factor_percent"]].drop_duplicates("company_reg_number").copy()
    return b.merge(f, on="company_reg_number", how="left")


def _build_account_monthly(credit_account: pd.DataFrame) -> pd.DataFrame:
    c = credit_account.dropna(subset=["company_reg_number", "ca_start_date"]).copy()
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
    g["ca_cum_new_accounts"] = g.groupby("company_reg_number")["ca_monthly_new_accounts"].cumsum()
    g["ca_cum_amount"] = g.groupby("company_reg_number")["ca_monthly_amount"].cumsum()
    g["ca_cum_payin"] = g.groupby("company_reg_number")["ca_monthly_payin"].cumsum()
    g["ca_cum_payout"] = g.groupby("company_reg_number")["ca_monthly_payout"].cumsum()
    g["ca_latest_rev_ratio"] = g.groupby("company_reg_number")["ca_latest_rev_ratio"].ffill()
    g["ca_latest_cost_ratio"] = g.groupby("company_reg_number")["ca_latest_cost_ratio"].ffill()
    return g[
        [
            "company_reg_number",
            "month_end",
            "ca_cum_new_accounts",
            "ca_cum_amount",
            "ca_cum_payin",
            "ca_cum_payout",
            "ca_latest_rev_ratio",
            "ca_latest_cost_ratio",
        ]
    ].copy()


def _merge_asof_company(left: pd.DataFrame, right: pd.DataFrame, right_time_col: str = "month_end") -> pd.DataFrame:
    if right.empty:
        return left
    l = left.copy()
    r = right.copy()
    l["month_end"] = pd.to_datetime(l["month_end"], errors="coerce")
    r[right_time_col] = pd.to_datetime(r[right_time_col], errors="coerce")
    l = l.dropna(subset=["company_reg_number", "month_end"]).copy()
    r = r.dropna(subset=["company_reg_number", right_time_col]).copy()
    l["company_reg_number"] = pd.to_numeric(l["company_reg_number"], errors="coerce").astype("int64")
    r["company_reg_number"] = pd.to_numeric(r["company_reg_number"], errors="coerce").astype("int64")

    payload = [c for c in r.columns if c not in {"company_reg_number", right_time_col}]
    out = []
    for cid, lg in l.groupby("company_reg_number", sort=False):
        lg = lg.sort_values("month_end")
        rg = r[r["company_reg_number"] == cid].sort_values(right_time_col)
        if rg.empty:
            for c in payload:
                lg[c] = np.nan
            out.append(lg)
            continue
        rgp = rg[[right_time_col] + payload]
        if right_time_col == "month_end":
            m = pd.merge_asof(lg, rgp, on="month_end", direction="backward")
        else:
            m = pd.merge_asof(lg, rgp, left_on="month_end", right_on=right_time_col, direction="backward")
            if right_time_col in m.columns:
                m = m.drop(columns=[right_time_col])
        out.append(m)
    return pd.concat(out, ignore_index=True)


def build_timevarying_augmented(loan_augmented: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for row in loan_augmented.itertuples(index=False):
        start_date = row.loan_start_date.normalize()
        end_date = (row.event_date_augmented if row.event_augmented == 1 else row.obs_end_date).normalize()
        if end_date < start_date:
            end_date = start_date
        cur = start_date
        while cur <= end_date:
            interval_end = min(_month_end(cur), end_date)
            start_t = (cur - start_date).days
            stop_t = (interval_end - start_date).days + 1
            if stop_t <= start_t:
                stop_t = start_t + 1

            interval_event = 0
            if row.event_augmented == 1 and pd.notna(row.event_date_augmented):
                e = row.event_date_augmented.normalize()
                if cur <= e <= interval_end:
                    interval_event = 1

            rows.append(
                {
                    "loan_key": row.loan_key,
                    "company_reg_number": row.company_reg_number,
                    "industry_group": row.industry_group,
                    "loan_start_date": start_date,
                    "obs_end_date": row.obs_end_date.normalize(),
                    "event_date_augmented": row.event_date_augmented.normalize() if pd.notna(row.event_date_augmented) else pd.NaT,
                    "event_type_augmented": row.event_type_augmented,
                    "event_hard_default": row.event_hard_default,
                    "event_distress": row.event_distress,
                    "event_augmented_loan": row.event_augmented,
                    "event_augmented_interval": interval_event,
                    "start": float(start_t),
                    "stop": float(stop_t),
                    "month_end": _month_end(interval_end),
                }
            )
            if interval_end >= end_date:
                break
            cur = interval_end + pd.Timedelta(days=1)

    iv = pd.DataFrame(rows)

    loan_cols = [
        "loan_key",
        "loan_original_amount",
        "interest",
        "loan_number_of_missed_payments",
        "total_time_payments_late",
        "is_overdraft",
        "is_term_loan",
        "duration_augmented_days",
        "behavior_distress_flag",
        "origination_credit_distress_flag",
        "business_failure_score",
        "credit_report_credit_score",
        "missed_and_late_payments_last_five_years",
        "credit_report_negative_items",
        "payment_index",
        "ratio_debt_to_revenue",
        "credit_report_total_indebtedness",
        "credit_report_good_standing_items",
    ]
    iv = iv.merge(loan_augmented[loan_cols], on="loan_key", how="left")

    static_feats = _build_static_business_features(tables["businesses"], tables["factoring"])
    iv = iv.merge(static_feats, on="company_reg_number", how="left")

    macro = _build_macro_monthly(tables["macro"])
    iv = iv.merge(macro, on="month_end", how="left")
    for c in ["uk_interest_rate", "uk_cpi", "uk_gdp", "uk_interest_rate_diff", "uk_cpi_diff", "uk_gdp_yoy"]:
        if c in iv.columns:
            iv[c] = iv[c].ffill().bfill()

    account_m = _build_account_monthly(tables["credit_account"])
    iv = _merge_asof_company(iv, account_m, right_time_col="month_end")

    iv["loan_age_days"] = iv["stop"]
    iv["log_loan_amount"] = np.log1p(iv["loan_original_amount"].clip(lower=0))
    iv["log_ca_cum_amount"] = np.log1p(pd.to_numeric(iv["ca_cum_amount"], errors="coerce").clip(lower=0))
    iv["log_ca_cum_payin"] = np.log1p(pd.to_numeric(iv["ca_cum_payin"], errors="coerce").clip(lower=0))
    iv["log_ca_cum_payout"] = np.log1p(pd.to_numeric(iv["ca_cum_payout"], errors="coerce").clip(lower=0))
    iv["log_factor_amount"] = np.log1p(pd.to_numeric(iv["factor_amount"], errors="coerce").clip(lower=0))

    iv = iv.sort_values(["loan_key", "start"]).reset_index(drop=True)
    return iv


def build_auxiliary_business_dataset(tables: Dict[str, pd.DataFrame], cfg: AugmentConfig) -> pd.DataFrame:
    b = tables["businesses"].copy()
    cr = _latest_credit_snapshot(tables["credit_rating"], cfg.snapshot_date)
    f = tables["factoring"][["company_reg_number", "factor_amount", "factor_percent"]].drop_duplicates("company_reg_number")
    ca = tables["credit_account"].copy()
    ca = ca.dropna(subset=["company_reg_number"]).copy()
    ca_stats = (
        ca.groupby("company_reg_number")
        .agg(
            ca_total_accounts=("number_of_accounts", "max"),
            ca_avg_rev_ratio=("rev_ratio", "mean"),
            ca_avg_cost_ratio=("cost_ratio", "mean"),
            ca_total_payin=("pay_in_amount", "sum"),
            ca_total_payout=("pay_out_amount", "sum"),
        )
        .reset_index()
    )

    cols = [
        "company_reg_number",
        "primary_sector",
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
    aux = b[cols].drop_duplicates("company_reg_number").copy()
    aux = aux.merge(cr, on="company_reg_number", how="left")
    aux = aux.merge(f, on="company_reg_number", how="left")
    aux = aux.merge(ca_stats, on="company_reg_number", how="left")

    aux["industry_group"] = aux["primary_sector"].fillna("Unknown").astype(str).str.strip()
    aux["liquidity_ratio"] = _safe_ratio(aux["current_assets"], aux["current_liabilities"])
    aux["leverage_ratio"] = _safe_ratio(
        aux["current_liabilities"].fillna(0) + aux["long_term_liabilities"].fillna(0),
        aux["capital_and_reserves"],
    )
    aux["profit_margin_proxy"] = _safe_ratio(aux["2019_revenue"] - aux["costs"], aux["2019_revenue"])
    aux["receivable_turnover_proxy"] = _safe_ratio(aux["2019_revenue"], aux["accounts_receivable"])
    aux["log_factor_amount"] = np.log1p(pd.to_numeric(aux["factor_amount"], errors="coerce").clip(lower=0))
    aux["log_ca_total_payin"] = np.log1p(pd.to_numeric(aux["ca_total_payin"], errors="coerce").clip(lower=0))
    aux["log_ca_total_payout"] = np.log1p(pd.to_numeric(aux["ca_total_payout"], errors="coerce").clip(lower=0))

    # Strict auxiliary distress label (for pretraining/risk representation)
    aux["aux_risk_label"] = (
        (aux["business_failure_score"].fillna(-1) >= 85)
        | (aux["credit_report_credit_score"].fillna(1000) <= 480)
        | (aux["missed_and_late_payments_last_five_years"].fillna(0) >= 6)
        | (aux["credit_report_negative_items"].fillna(0) >= 7)
    ).astype(int)
    return aux


def save_outputs(
    loan_augmented: pd.DataFrame,
    timevarying_augmented: pd.DataFrame,
    auxiliary_business: pd.DataFrame,
    cfg: AugmentConfig,
) -> None:
    data_dir = cfg.data_dir
    loan_augmented.to_csv(data_dir / cfg.output_loan_augmented, index=False, encoding="utf-8-sig")
    timevarying_augmented.to_csv(data_dir / cfg.output_timevarying_augmented, index=False, encoding="utf-8-sig")
    auxiliary_business.to_csv(data_dir / cfg.output_auxiliary, index=False, encoding="utf-8-sig")

    report_lines = [
        "Data Augmentation Summary (Real-Signal Expansion)",
        f"snapshot_date={cfg.snapshot_date.date()}",
        "",
        f"[Loan base] rows={len(loan_augmented)}",
        f"hard_default_events={int(loan_augmented['event_hard_default'].sum())}",
        f"distress_events_added={int(loan_augmented['event_distress'].sum())}",
        f"augmented_total_events={int(loan_augmented['event_augmented'].sum())}",
        f"event_type_counts={loan_augmented['event_type_augmented'].value_counts().to_dict()}",
        "",
        f"[Loan time-varying augmented] rows={len(timevarying_augmented)}",
        f"loan_count={timevarying_augmented['loan_key'].nunique()}",
        f"interval_events={int(timevarying_augmented['event_augmented_interval'].sum())}",
        "",
        f"[Auxiliary enterprise dataset] rows={len(auxiliary_business)}",
        f"company_count={auxiliary_business['company_reg_number'].nunique()}",
        f"aux_risk_positive={int(auxiliary_business['aux_risk_label'].sum())}",
        f"aux_risk_rate={auxiliary_business['aux_risk_label'].mean():.4f}",
        "",
        "files:",
        f"- {cfg.output_loan_augmented}",
        f"- {cfg.output_timevarying_augmented}",
        f"- {cfg.output_auxiliary}",
    ]
    (data_dir / cfg.output_report).write_text("\n".join(report_lines), encoding="utf-8-sig")


def main() -> None:
    data_dir = Path(r"E:\my model\SME_model\SME - Synthetic UK Businesses financial statistics")
    cfg = AugmentConfig(data_dir=data_dir)
    tables = load_tables(cfg.data_dir)
    loan_augmented = build_loan_augmented(tables, cfg)
    timevarying_augmented = build_timevarying_augmented(loan_augmented, tables)
    auxiliary_business = build_auxiliary_business_dataset(tables, cfg)
    save_outputs(loan_augmented, timevarying_augmented, auxiliary_business, cfg)

    print("=== Augmentation Completed ===")
    print(f"Loan rows: {len(loan_augmented)}")
    print(f"Hard default events: {int(loan_augmented['event_hard_default'].sum())}")
    print(f"Added distress events: {int(loan_augmented['event_distress'].sum())}")
    print(f"Augmented total events: {int(loan_augmented['event_augmented'].sum())}")
    print(f"Time-varying rows: {len(timevarying_augmented)}")
    print(f"Auxiliary enterprise rows: {len(auxiliary_business)}")


if __name__ == "__main__":
    main()

