"""Models page: ranking, the effect of class imbalance remedies, calibration and the cost of transparency."""

import streamlit as st

import charts
import labels
import loaders

TOP_HEIGHT = 450
REMEDY_HEIGHT = 450
CALIBRATION_HEIGHT = 550

st.title("Models")
st.caption(
    "Five models compared with stratified 5-fold cross-validation on 80% of the loans, then scored once on the "
    "held-out 20%. Logistic regression, LDA, the decision tree and gradient boosting are also trained with six class "
    "imbalance remedies."
)


def readable(table):
    """Replace model/remedy codes with display names."""
    return table.assign(remedy=table["remedy"].map(labels.REMEDIES))


test = readable(loaders.artifact("test_results.csv"))
cv = readable(loaders.artifact("cv_results.csv"))
default_rate = loaders.test_default_rate()
baseline = test[test["remedy"] == "No remedy"]
remedies = test[~test["remedy"].isin(["No remedy", "Yeo-Johnson transform"])]

cv_change = cv.merge(cv[cv["remedy"] == "No remedy"][["model", "ap"]], on="model", suffixes=("", "_baseline"))
cv_change = cv_change[cv_change["remedy"].isin(remedies["remedy"].unique())].assign(ap_change=lambda d: d["ap"] - d["ap_baseline"])
best = baseline.loc[baseline["gini"].idxmax()]
scorecard = baseline[baseline["model"] == "Scorecard"].iloc[0]

with st.container(horizontal=True):
    st.metric("Best Gini", f"{best['gini']:.2f}", border=True, height="stretch", help=f"{best['model']}, test set.")
    st.metric("Scorecard Gini", f"{scorecard['gini']:.2f}", border=True, height="stretch",
              help="Transparent WoE scorecard, test set.")
    st.metric("Largest ranking gain from a remedy", f"{cv_change['ap_change'].max():+.3f}", border=True, height="stretch",
              help="Change in cross-validated average precision against the same model without remedy.")
    st.metric("Highest mean PD with a remedy", f"{remedies['mean_pd'].max():.1%}", border=True, height="stretch",
              help=f"Against an actual default rate of {default_rate:.1%} on the test set.")

st.header("Which model ranks risk best?", divider="gray")
with st.container(horizontal=True, vertical_alignment="bottom", gap="large"):
    shown = st.pills("Models", labels.MODELS, selection_mode="multi",
                     default=["Gradient Boosting", "Scorecard", "Logistic Regression"], key="curve_models")
    curve = st.segmented_control("Curve", ["ROC", "Precision-recall"], default="ROC", required=True, key="curve")

if not shown:
    st.info("Select at least one model.", icon=":material/touch_app:")
else:
    if curve == "ROC":
        data = loaders.artifact("roc.csv")
        chart = charts.curves(data[(data["remedy"] == "baseline") & data["model"].isin(shown)],
                              "fpr", "tpr", "False positive rate", "True positive rate", diagonal=True)
    else:
        data = loaders.artifact("precision_recall.csv")
        chart = charts.curves(data[(data["remedy"] == "baseline") & data["model"].isin(shown)],
                              "recall", "precision", "Recall (defaults detected)", "Precision", diagonal=False)
    gini_col, curve_col = st.columns(2, gap="large")
    with gini_col:
        st.altair_chart(charts.gini_bars(baseline[baseline["model"].isin(shown)]), height=TOP_HEIGHT,
                        alt="Test Gini per model without remedy, highest for gradient boosting")
        st.caption("Gini = 2 x ROC AUC - 1: 0 for random ranking, 1 for perfect ranking. Models trained without remedy.")
    with curve_col:
        st.altair_chart(chart, height=TOP_HEIGHT, alt=f"{curve} curves on the test set for the selected models")
        st.caption(f"{curve} curves on the test set.")

st.header("Do class imbalance remedies help?", divider="gray")
st.markdown(
    "Defaults are outnumbered 1 to 4, which is often addressed by reweighting or resampling the training data. "
    "**The remedies leave the ranking of loans unchanged, but they inflate predicted PDs.**"
)
rank_col, pd_col = st.columns(2, gap="large")
with rank_col:
    st.altair_chart(
        charts.remedy_strip(cv_change, "ap_change", "Change in CV average precision vs. no remedy", 0, "No change"),
        height=REMEDY_HEIGHT, alt="Remedies change cross-validated average precision by about 0.01 at most",
    )
    st.caption("Ranking: every remedy stays within about 0.01 of the model without remedy.")
with pd_col:
    st.altair_chart(
        charts.remedy_strip(remedies, "mean_pd", "Mean predicted PD (test set)", default_rate, f"Actual default rate {default_rate:.1%}"),
        height=REMEDY_HEIGHT, alt="Remedies raise the mean predicted PD well above the actual default rate",
    )
    st.caption(f"Calibration: the dashed line is the actual default rate ({default_rate:.1%}).")

st.subheader("Calibration curves")
model = st.segmented_control("Model", [m for m in labels.MODELS if m != "Scorecard"], default="Gradient Boosting",
                             required=True, key="calibration_model")
calibration = readable(loaders.artifact("calibration.csv"))
calibration = calibration[(calibration["model"] == model) & (calibration["remedy"] != "Yeo-Johnson transform")]
st.altair_chart(charts.calibration(calibration), height=CALIBRATION_HEIGHT, alt=f"Calibration curves of {model} per remedy against the diagonal")
st.caption("10 groups of equal size on the test set. On the diagonal, a predicted PD of 30% means 30% of loans default. "
           "Curves below the diagonal overstate PD.")

st.header("The cost of transparency", divider="gray")
st.markdown(
    "Banks usually deploy scorecards because regulators can check every effect and declined applicants can be told why. "
    "The scorecard ranks and calibrates better than plain logistic regression, as binning captures non-linear effects, "
    "but gradient boosting, which also models interactions between features, stays clearly ahead."
)
columns = {"model": "Model", "gini": "Gini", "ks": "KS", "ap": "Average precision", "brier": "Brier score", "mean_pd": "Mean PD"}
comparison = baseline[list(columns)].sort_values("gini", ascending=False).rename(columns=columns)
st.dataframe(
    comparison, hide_index=True,
    column_config={name: st.column_config.NumberColumn(format="%.3f") for name in ["Gini", "KS", "Average precision", "Brier score"]}
    | {"Mean PD": st.column_config.NumberColumn(format="percent")},
    alt="Test set ranking and calibration of the five models without remedy",
)

with st.expander("All results (test set)", icon=":material/table:"):
    st.dataframe(test, hide_index=True, alt="Test set results for every model and remedy")
