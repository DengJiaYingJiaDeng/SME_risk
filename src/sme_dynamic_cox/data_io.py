from __future__ import annotations

from pathlib import Path
from typing import Dict
import warnings

import pandas as pd


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "cp1252", "latin1"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
    raise RuntimeError(f"Failed to read {path} with fallback encodings.") from last_error


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _to_datetime(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed", dayfirst=False)
                needs_fallback = parsed.isna() & df[col].notna()
                if needs_fallback.any():
                    parsed_fallback = pd.to_datetime(
                        df.loc[needs_fallback, col],
                        errors="coerce",
                        format="mixed",
                        dayfirst=True,
                    )
                    parsed.loc[needs_fallback] = parsed_fallback
            df[col] = parsed
    return df


def load_all_tables(data_dir: Path) -> Dict[str, pd.DataFrame]:
    required = {
        "businesses": "Businesses.csv",
        "loan": "Loan.csv",
        "credit_rating": "Credit_Rating.csv",
        "credit_account": "Credit_Account_History.csv",
        "credit_card": "Credit_Card_History.csv",
        "factoring": "Factoring.csv",
        "macro": "Combined_macro_data.csv",
    }
    optional = {"covid": "COVID.csv", "account_receivable": "Account_Receivable.csv"}

    tables: Dict[str, pd.DataFrame] = {}
    for key, file_name in required.items():
        file_path = data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file: {file_path}")
        tables[key] = _read_csv_with_fallback(file_path)

    for key, file_name in optional.items():
        file_path = data_dir / file_name
        tables[key] = _read_csv_with_fallback(file_path) if file_path.exists() else pd.DataFrame()

    tables["loan"] = _to_datetime(
        tables["loan"],
        ["loan_start_date", "loan_date_due_to_close", "loan_default_date", "loan_satisfaction_date"],
    )
    tables["loan"] = _to_numeric(
        tables["loan"],
        [
            "loan_original_amount",
            "interest",
            "loan_number_of_missed_payments",
            "loan_amount_outstanding_including_future_interest",
            "loan",
            "overdraft",
        ],
    )

    tables["businesses"] = _to_datetime(tables["businesses"], ["incorporation_date", "dissolved_on", "filing_date"])
    tables["businesses"] = _to_numeric(
        tables["businesses"],
        [
            "company_reg_number",
            "women_owned",
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
            "number_of_officers",
        ],
    )

    tables["credit_rating"] = _to_datetime(tables["credit_rating"], ["credit_report_date"])
    tables["credit_rating"] = _to_numeric(
        tables["credit_rating"],
        [
            "company_reg_number",
            "credit_report_requests_to_view_last_five_years",
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
        ],
    )

    tables["credit_account"] = _to_datetime(tables["credit_account"], ["ca_start_date"])
    tables["credit_account"] = _to_numeric(
        tables["credit_account"],
        [
            "company_reg_number",
            "number_of_accounts",
            "current_acc_fraction",
            "total_amount",
            "amount",
            "rev_ratio",
            "cost_ratio",
            "pay_in_amount",
            "pay_out_amount",
        ],
    )

    tables["credit_card"] = _to_datetime(tables["credit_card"], ["cc_start_date"])
    tables["credit_card"] = _to_numeric(
        tables["credit_card"],
        [
            "company_reg_number",
            "cc_agreed_limit",
            "cc_balance_limit_ratio",
            "missed_payments_number",
        ],
    )

    tables["factoring"] = _to_numeric(
        tables["factoring"],
        ["company_reg_number", "revenue_2019", "factor_amount", "factor_percent"],
    )

    tables["macro"] = _to_datetime(tables["macro"], ["calendar_month"])
    tables["macro"] = _to_numeric(tables["macro"], ["uk_interest_rate", "uk_cpi", "uk_gdp"])

    for table in tables.values():
        if "company_reg_number" in table.columns:
            table["company_reg_number"] = pd.to_numeric(table["company_reg_number"], errors="coerce").astype("Int64")

    return tables
