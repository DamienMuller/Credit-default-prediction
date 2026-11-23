"""Explain a scorecard score: points earned per feature and the features that lower it most."""

import pandas as pd


def bin_label(bin_):
    """Readable bin label: categorical bins are arrays of categories, numeric bins are interval strings."""
    return bin_ if isinstance(bin_, str) else ", ".join(bin_)


def points_breakdown(scorecard, applicant):
    """Points per feature for one applicant, sorted by points lost against the best bin of each feature.

    Parameters
    ----------
    scorecard : optbinning.Scorecard
        Fitted scorecard.
    applicant : pandas.DataFrame
        One row with the model features.

    Returns
    -------
    pandas.DataFrame
        Feature, bin, points, max_points and points_lost; points sum to the applicant's score.
    """
    table = scorecard.table(style="detailed")
    table = table[table["Count"] > 0]
    bins = scorecard.binning_process_.transform(applicant, metric="indices").iloc[0]
    rows = []
    for feature, index in bins.items():
        feature_table = table[table["Variable"] == feature]
        rows.append({
            "feature": feature,
            "bin": bin_label(feature_table.loc[index, "Bin"]),
            "points": feature_table.loc[index, "Points"],
            "max_points": feature_table["Points"].max(),
        })
    breakdown = pd.DataFrame(rows)
    breakdown["points_lost"] = breakdown["max_points"] - breakdown["points"]
    return breakdown.sort_values("points_lost", ascending=False, ignore_index=True)
