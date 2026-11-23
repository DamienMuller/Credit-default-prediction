"""Data insights page: default rates by category and along numeric features, and information value per feature."""

import pandas as pd
import streamlit as st

import charts
import labels
import loaders
import stats

CHART_HEIGHT = 420
CATEGORICAL = {
    "Loan grade": "loan_grade",
    "Home ownership": "person_home_ownership",
    "Loan intent": "loan_intent",
    "Previous default": "cb_person_default_on_file",
}
NUMERIC = {
    "Loan-to-income ratio": ("loan_percent_income", ".2f"),
    "Annual income": ("person_income", "$,.0f"),
    "Interest rate (%)": ("loan_int_rate", ".1f"),
    "Loan amount": ("loan_amnt", "$,.0f"),
}
IV_BANDS = [(0.02, "Not useful"), (0.1, "Weak"), (0.3, "Medium"), (0.5, "Strong"), (float("inf"), "Very strong")]

st.title("Data insights")
st.caption("32,581 personal loans from a public Datacamp dataset; 7 loans with an implausible age or employment length are removed.")

loans = loaders.loans()
average = loans["loan_status"].mean()

with st.container(horizontal=True):
    st.metric("Loans", f"{len(loans):,}", border=True, height="stretch")
    st.metric("Default rate", f"{average:.1%}", border=True, height="stretch", help="Share of loans that defaulted.")
    st.metric("Median annual income", f"${loans['person_income'].median():,.0f}", border=True, height="stretch")
    st.metric("Median loan amount", f"${loans['loan_amnt'].median():,.0f}", border=True, height="stretch")

st.header("Feature Analysis", divider="gray")
category_col, numeric_col = st.columns(2, gap="large")
with category_col:
    name = st.segmented_control("Categorical variables", list(CATEGORICAL), default="Loan grade", required=True, key="category")
    rates = stats.default_rate_by(loans, CATEGORICAL[name])
    rates["label"] = rates["category"].map(lambda value: {"Y": "Yes", "N": "No"}.get(value, labels.category(value)))
    sort = list(rates["label"]) if name == "Loan grade" else "-y"
    st.altair_chart(charts.rate_bars(rates, name, average, sort), height=CHART_HEIGHT,
                    alt=f"Default rate by {name.lower()} against the average default rate")
    st.caption("Dashed line: average default rate. Hover over a bar for the number of loans.")
with numeric_col:
    name = st.segmented_control("Numerical variables", list(NUMERIC), default="Loan-to-income ratio", required=True, key="numeric")
    feature, value_format = NUMERIC[name]
    st.altair_chart(charts.rate_line(stats.default_rate_by_quantile(loans, feature), name, value_format, average),
                    height=CHART_HEIGHT, alt=f"Default rate along {name.lower()}")
    st.caption("Each point is a group of about 5% of the loans, placed at its median value.")

st.markdown(
    "- **Loan-to-income is the strongest single driver.** The default rate stays below 20% up to a ratio of about 0.25, "
    "then jumps to about 70% from 0.31 on, the threshold the scorecard's binning also found.\n"
    "- **Grades D to G default far more often** than A to C. Grade and interest rate are set by the lender's own pricing, "
    "so they already carry its risk assessment.\n"
    "- **Renters default more than owners and mortgage holders,** and low incomes are much riskier than high ones."
)

st.header("What drives risk?", divider="gray")
iv = loaders.artifact("scorecard_iv.csv")
iv = iv.assign(
    feature=iv["name"].map(labels.FEATURES),
    strength=pd.cut(iv["iv"], [0] + [edge for edge, _ in IV_BANDS], labels=[band for _, band in IV_BANDS]),
)
iv_col, text_col = st.columns([3, 2], gap="large")
with iv_col:
    st.altair_chart(charts.iv_bars(iv), height=CHART_HEIGHT, alt="Information value per feature, highest for the loan-to-income ratio")
with text_col:
    st.markdown(
        "**Information value (IV)** measures how well a feature separates defaults from non-defaults after optimal binning. "
        "It is the standard feature screening measure for credit scorecards.\n\n"
        "| IV | Predictive power |\n|---|---|\n| below 0.02 | Not useful |\n| 0.02 to 0.1 | Weak |\n"
        "| 0.1 to 0.3 | Medium |\n| 0.3 to 0.5 | Strong |\n| above 0.5 | Very strong |\n\n"
        "Age and credit history length are not useful and are left out of the scorecard. The previous-default flag "
        "is also left out, as its effect overlaps with the loan grade."
    )

with st.expander("Data quality", icon=":material/fact_check:"):
    missing = loans.isna().mean()
    st.markdown(
        f"- Interest rate is missing for {missing['loan_int_rate']:.1%} of loans and employment length for "
        f"{missing['person_emp_length']:.1%}. Models impute the mean; the scorecard gives missing values their own bin.\n"
        "- 7 loans with an age or employment length above 100 years are removed.\n"
        "- The data has no application dates, so all results come from a random (not time-based) split."
    )
