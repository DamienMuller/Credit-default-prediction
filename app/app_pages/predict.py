"""Predict PD page: client form, PDs from the champion and the scorecard, and scorecard reasons."""

import altair as alt
import pandas as pd
import streamlit as st

import labels
import loaders
from credit_risk.data import CATEGORICAL, NUMERIC
from credit_risk.explain import points_breakdown

st.title("Predict PD")
st.caption(
    "Probability of default for a new client, from the champion model (gradient boosting) "
    "and the transparent challenger (WoE scorecard)."
)

loans = loaders.loans()
form_col, result_col = st.columns([2, 3], gap="large")

with form_col:
    with st.container(border=True):
        st.subheader("Applicant", divider="gray")
        age = st.number_input("Age", min_value=18, max_value=100, value=26, key="age")
        income = st.number_input("Annual income ($)", min_value=4000, max_value=2_000_000, value=55_000, step=1000, key="income")
        home = st.segmented_control(
            "Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], default="RENT", required=True, format_func=labels.category, key="home"
        )
        emp_length = st.number_input("Employment length (years)", min_value=0, max_value=60, value=4, key="emp_length")
        history = st.number_input("Credit history length (years)", min_value=2, max_value=30, value=4, key="history")
        previous_default = st.toggle("Previous default on file", key="previous_default")

    with st.container(border=True):
        st.subheader("Loan", divider="gray")
        intent = st.selectbox("Loan intent", sorted(loans["loan_intent"].unique()), format_func=labels.category, key="intent")
        amount = st.number_input("Loan amount ($)", min_value=500, max_value=35_000, value=8000, step=500, key="amount")
        grade = st.segmented_control("Loan grade", list("ABCDEFG"), default="A", required=True, key="grade")
        rate = st.number_input("Interest rate (%)", min_value=5.0, max_value=24.0, value=11.0, step=0.25, key="rate")
        st.caption("Grade and interest rate are set by the lender's own pricing of the loan.")

applicant = pd.DataFrame([{
    "person_age": age,
    "person_income": income,
    "person_emp_length": emp_length,
    "loan_amnt": amount,
    "loan_int_rate": rate,
    "loan_percent_income": round(amount / income, 2),
    "cb_person_cred_hist_length": history,
    "person_home_ownership": home,
    "loan_intent": intent,
    "loan_grade": grade,
    "cb_person_default_on_file": "Y" if previous_default else "N",
}])[NUMERIC + CATEGORICAL]

boosting = loaders.pd_model("boosting")
scorecard_model = loaders.pd_model("scorecard")
scorecard = scorecard_model.estimator_
pd_boosting = boosting.predict_proba(applicant)[0, 1]
pd_scorecard = scorecard_model.predict_proba(applicant)[0, 1]
score = scorecard.score(applicant)[0]
breakdown = points_breakdown(scorecard, applicant)

with result_col:
    with st.container(horizontal=True):
        st.metric("PD, gradient boosting", f"{pd_boosting:.1%}", border=True, height="stretch",
                  help="Champion model: best ranking and calibration on the test set (Gini 0.88).")
        st.metric("PD, scorecard", f"{pd_scorecard:.1%}", border=True, height="stretch",
                  help="Transparent challenger: optimal binning, weight of evidence and logistic regression (Gini 0.76).")
        st.metric("Score", f"{score:.0f}", border=True, height="stretch",
                  help="600 points at good:bad odds of 50:1; 20 more points double the odds.")

    flagged = boosting.predict(applicant)[0] == 1
    st.badge(
        "Above the risk cutoff" if flagged else "Below the risk cutoff",
        icon=":material/warning:" if flagged else ":material/check_circle:",
        color="red" if flagged else "green",
        help=f"Gradient boosting cutoff of {boosting.best_threshold_:.1%}, tuned to maximize the F-score. "
             f"Average default rate: {loans['loan_status'].mean():.1%}.",
    )
    st.caption(f"Loan-to-income ratio: {applicant['loan_percent_income'].iloc[0]:.2f}")

    st.subheader("Why this score", divider="gray")
    reasons = breakdown[breakdown["points_lost"] > 0.5].head(3)
    if reasons.empty:
        st.success("Every feature is in its best bin.", icon=":material/verified:")
    for row in reasons.itertuples():
        st.markdown(f"**{labels.FEATURES[row.feature]}** ({labels.bin_text(row.feature, row.bin)}): {row.points_lost:.0f} points below the best bin")

    chart_data = breakdown.assign(feature=breakdown["feature"].map(labels.FEATURES), bin=[labels.bin_text(f, b) for f, b in zip(breakdown["feature"], breakdown["bin"])])
    base = alt.Chart(chart_data).encode(y=alt.Y("feature:N", sort=None, title=None))
    best = base.mark_bar(color="lightgray").encode(x=alt.X("max_points:Q", title="Points"))
    earned = base.mark_bar().encode(
        x="points:Q",
        tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("bin:N", title="Bin"),
                 alt.Tooltip("points:Q", title="Points", format=".0f"), alt.Tooltip("max_points:Q", title="Best bin", format=".0f")],
    )
    st.altair_chart(best + earned, alt="Points earned per feature against the best bin, sorted by points lost")
    st.caption("Grey: points of the best bin of each feature. Age, credit history length and the previous-default flag are not in the scorecard.")
