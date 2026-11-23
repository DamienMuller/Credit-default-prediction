"""Executive overview: the project's most decision-relevant findings."""

import altair as alt
import streamlit as st

import charts
import labels
import loaders

st.badge("CREDIT RISK · Research Project", icon=":material/verified_user:", color="blue")
st.title("Class imbalance in PD modelling: why resampling doesn't help, and what does.")
st.markdown(
    "A hands-on comparison of **five credit risk models** across **32k+ personal loans**. "
    "See what improves default detection, what breaks probability calibration, and where an "
    "interpretable scorecard still earns its place."
)

test = loaders.artifact("test_results.csv")
baseline = test[test["remedy"] == "baseline"].copy()
loans = loaders.loans()
default_rate = loaders.test_default_rate()
champion = baseline.loc[baseline["gini"].idxmax()]
boosting = baseline[baseline["model"] == "Gradient Boosting"].iloc[0]
scorecard = baseline[baseline["model"] == "Scorecard"].iloc[0]

with st.grid(4, min_column_width=170, gap="small"):
    st.metric("Loans analyzed", f"{len(loans):,}", border=True, height="stretch")
    st.metric("Held-out default rate", f"{default_rate:.1%}", border=True, height="stretch")
    st.metric("Best test Gini", f"{champion['gini']:.2f}", border=True, height="stretch", help=f"{champion['model']} · 20% held-out test set")
    st.metric("Champion average precision", f"{boosting['ap']:.3f}", border=True, height="stretch", help="Test set; cross-validation average was 0.900 ± 0.005.")

st.subheader("The model choice is a trade-off")
chart_col, story_col = st.columns([1.35, 1], gap="large")
with chart_col:
    with st.container(border=True):
        st.markdown("**Ranking power × probability accuracy**")
        st.caption("Held-out test set · upper right is better")
        comparison = baseline[["model", "gini", "brier"]].copy()
        points = alt.Chart(comparison).mark_circle(size=180, opacity=0.9).encode(
            x=alt.X("gini:Q", title="Gini · ranking power", scale=alt.Scale(domain=[0.68, 0.92])),
            y=alt.Y("brier:Q", title="Brier score · lower is better", scale=alt.Scale(domain=[0.12, 0.045])),
            color=alt.Color("model:N", title="Model", scale=alt.Scale(domain=labels.MODELS)),
            tooltip=["model:N", alt.Tooltip("gini:Q", format=".3f"), alt.Tooltip("brier:Q", format=".3f")],
        )
        names = points.mark_text(dy=-16, fontSize=12).encode(text="model:N", color=alt.value("#d9e2f0"))
        st.altair_chart(points + names, height=330, alt="Held-out comparison of model ranking power and probability accuracy; gradient boosting ranks best while the scorecard remains interpretable.")
        st.caption("Gini measures separation; Brier score measures probability error. These are different jobs, so compare both before choosing a model.")

with story_col:
    with st.container(border=True, height="stretch"):
        st.markdown(":material/bolt: **Champion · gradient boosting**")
        st.metric("Gini", f"{boosting['gini']:.3f}", border=False, help="Test set: 0.882. Cross-validation average precision: 0.900 ± 0.005.")
        st.markdown(
            "Ranks default risk best, with a **0.892 average precision** and **0.057 Brier score**. "
            "Its threshold is tuned on training folds, not the test set."
        )
        st.divider()
        st.markdown(":material/visibility: **Transparent challenger · scorecard**")
        st.metric("Gini", f"{scorecard['gini']:.3f}", border=False, help="Test set: 0.764. Eight retained scorecard features.")
        st.markdown(
            "Gives up **0.118 Gini** versus boosting, while keeping risk effects inspectable "
            "through optimal bins, weight of evidence and points."
        )

st.subheader("The finding that changes the modelling decision")
left, right = st.columns(2, gap="medium")
with left:
    with st.container(border=True, height="stretch"):
        st.markdown(":material/tune: **Resampling did not solve ranking**")
        st.markdown(
            "Across logistic regression, LDA, trees and boosting, class weights and resampling "
            "kept cross-validated average precision within roughly **0.01** of the no-remedy model. "
            "A more complicated training recipe did not create a meaningfully better ranker."
        )
with right:
    with st.container(border=True, height="stretch"):
        st.markdown(":material/warning: **But it shifted the PD scale**")
        st.markdown(
            f"The observed test default rate is **{default_rate:.1%}**. Some remedies pushed mean "
            "predicted default probability as high as **37.7%** for logistic regression. "
            "That matters when PD feeds pricing, provisions or capital."
        )

st.subheader("Explore the work")
with st.grid(2, min_column_width=210, gap="small"):
    with st.container(border=True, height="stretch"):
        st.markdown(":material/analytics: **Model performance**")
        st.caption("ROC and precision-recall curves, calibration, threshold metrics, and imbalance experiments.")
        st.page_link("app_pages/models.py", label="Compare models", icon=":material/arrow_forward:", width="stretch")
    with st.container(border=True, height="stretch"):
        st.markdown(":material/query_stats: **Portfolio insights**")
        st.caption("Default rates by grade, affordability, loan purpose, home ownership, and information value.")
        st.page_link("app_pages/insights.py", label="Explore the data", icon=":material/arrow_forward:", width="stretch")

st.caption("Research project · Datacamp loan dataset · stratified 80/20 split · thresholds selected with 5-fold stratified cross-validation")
