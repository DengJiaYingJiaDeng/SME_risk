from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    snapshot_date: str = "2024-02-29"
    test_size: float = 0.3
    min_train_events: int = 4
    min_test_events: int = 2
    penalizer: float = 0.25
    l1_ratio: float = 0.0
    warning_horizon_days: int = 90
    decision_horizon_days: int = 180
    min_samples_per_industry: int = 6
    loan_approval_gain: float = 1.0
    loan_default_loss: float = 4.0
    random_state: int = 42
    preferred_feature_order: List[str] = field(
        default_factory=lambda: [
            "log_loan_amount",
            "loan_number_of_missed_payments",
            "loan_age_days",
            "is_overdraft",
            "is_term_loan",
            "women_owned",
            "number_of_officers",
            "liquidity_ratio",
            "leverage_ratio",
            "profit_margin_proxy",
            "working_capital",
            "receivable_turnover_proxy",
            "factor_percent",
            "log_factor_amount",
            "uk_interest_rate_diff",
            "uk_cpi_diff",
            "uk_gdp_yoy",
            "credit_report_total_indebtedness",
            "payment_index",
            "credit_report_negative_items",
            "credit_report_good_standing_items",
            "business_failure_score",
            "credit_report_credit_score",
            "ratio_debt_to_revenue",
            "credit_signal_score",
            "ca_cum_new_accounts",
            "log_ca_cum_amount",
            "log_ca_cum_payin",
            "log_ca_cum_payout",
            "ca_latest_rev_ratio",
            "ca_latest_cost_ratio",
            "cc_cum_new_cards",
            "log_cc_cum_agreed_limit",
            "cc_latest_balance_limit_ratio",
            "cc_cum_missed_payments",
        ]
    )
