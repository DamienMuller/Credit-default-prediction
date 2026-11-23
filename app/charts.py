"""Altair charts shared by the app pages. Model colors follow labels.MODELS on every chart."""

import altair as alt
import pandas as pd

from labels import MODELS

MODEL_COLOR = alt.Color("model:N", title="Model", scale=alt.Scale(domain=MODELS))


def gini_bars(results):
    """Horizontal bars of test Gini per model, best first."""
    base = alt.Chart(results).encode(
        y=alt.Y("model:N", sort="-x", title=None),
        x=alt.X("gini:Q", title="Gini (test set)", scale=alt.Scale(domain=[0, 1])),
    )
    bars = base.mark_bar().encode(color=MODEL_COLOR.legend(None), tooltip=["model:N", alt.Tooltip("gini:Q", format=".3f")])
    text = base.mark_text(align="left", dx=4).encode(text=alt.Text("gini:Q", format=".2f"))
    return bars + text


def curves(data, x, y, x_title, y_title, diagonal):
    """ROC or precision-recall curves per model, with an optional random-model diagonal."""
    lines = alt.Chart(data).mark_line().encode(
        x=alt.X(f"{x}:Q", title=x_title), y=alt.Y(f"{y}:Q", title=y_title), color=MODEL_COLOR, order=f"{x}:Q"
    )
    if not diagonal:
        return lines
    reference = alt.Chart(pd.DataFrame({x: [0, 1], y: [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x=f"{x}:Q", y=f"{y}:Q")
    return reference + lines


def remedy_strip(data, field, title, reference, reference_label):
    """One dot per remedy and model, with a vertical reference line (e.g. no change, or the actual default rate)."""
    dots = alt.Chart(data).mark_circle(size=110, opacity=0.85).encode(
        x=alt.X(f"{field}:Q", title=title),
        y=alt.Y("model:N", sort=MODELS, title=None),
        color=alt.Color("remedy:N", title="Remedy"),
        tooltip=["model:N", "remedy:N", alt.Tooltip(f"{field}:Q", format=".3f")],
    )
    rule = alt.Chart(pd.DataFrame({field: [reference], "label": [reference_label]})).mark_rule(strokeDash=[4, 4], color="gray")
    rule = rule.encode(x=f"{field}:Q", tooltip="label:N")
    return rule + dots


def calibration(data):
    """Observed default rate against mean predicted PD per remedy; the model without remedy is drawn thicker."""
    lines = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("predicted_pd:Q", title="Mean predicted PD", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed_rate:Q", title="Observed default rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("remedy:N", title="Remedy"),
        strokeWidth=alt.condition(alt.datum.remedy == "No remedy", alt.value(3.5), alt.value(1.5)),
        tooltip=["remedy:N", alt.Tooltip("predicted_pd:Q", title="Predicted PD", format=".2f"),
                 alt.Tooltip("observed_rate:Q", title="Observed rate", format=".2f")],
    )
    diagonal = alt.Chart(pd.DataFrame({"predicted_pd": [0, 1], "observed_rate": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray")
    return diagonal.encode(x="predicted_pd:Q", y="observed_rate:Q") + lines


def rate_bars(data, title, average, sort):
    """Default rate per category with a dashed line at the average default rate."""
    bars = alt.Chart(data).mark_bar().encode(
        x=alt.X("label:N", sort=sort, title=title, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("default_rate:Q", title="Default rate", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("label:N", title=title), alt.Tooltip("default_rate:Q", title="Default rate", format=".1%"),
                 alt.Tooltip("loans:Q", title="Loans", format=",")],
    )
    return bars + average_rule(average)


def rate_line(data, title, value_format, average):
    """Default rate along a numeric feature (one point per group of equal size) with the average default rate."""
    line = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("value:Q", title=title, axis=alt.Axis(format=value_format)),
        y=alt.Y("default_rate:Q", title="Default rate", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("value:Q", title=f"{title} (median)", format=value_format),
                 alt.Tooltip("default_rate:Q", title="Default rate", format=".1%"), alt.Tooltip("loans:Q", title="Loans", format=",")],
    )
    return average_rule(average) + line


def average_rule(average):
    """Horizontal dashed line at the average default rate."""
    data = pd.DataFrame({"default_rate": [average], "label": [f"Average default rate {average:.1%}"]})
    return alt.Chart(data).mark_rule(strokeDash=[4, 4], color="gray").encode(y="default_rate:Q", tooltip="label:N")


def iv_bars(data):
    """Information value per feature, colored by its conventional strength band."""
    return alt.Chart(data).mark_bar().encode(
        y=alt.Y("feature:N", sort="-x", title=None),
        x=alt.X("iv:Q", title="Information value"),
        color=alt.Color("strength:N", title="Predictive power", sort=["Very strong", "Strong", "Medium", "Weak", "Not useful"]),
        tooltip=["feature:N", alt.Tooltip("iv:Q", title="Information value", format=".3f"), "strength:N"],
    )
