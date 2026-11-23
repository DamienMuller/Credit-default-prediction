"""Default-rate aggregations of the loan dataset for the insights page."""

import pandas as pd


def default_rate_by(loans, feature):
    """Default rate and number of loans per category of a feature."""
    return (
        loans.groupby(feature, observed=True)["loan_status"]
        .agg(default_rate="mean", loans="size")
        .reset_index()
        .rename(columns={feature: "category"})
    )


def default_rate_by_quantile(loans, feature, groups=20):
    """Default rate per group of equal size along a numeric feature, positioned at the group's median value."""
    group = pd.qcut(loans[feature], groups, duplicates="drop")
    return (
        loans.groupby(group, observed=True)
        .agg(value=(feature, "median"), default_rate=("loan_status", "mean"), loans=("loan_status", "size"))
        .reset_index(drop=True)
    )
