"""WoE scorecard: optimal monotonic binning, logistic regression on WoE values, points scaling."""

from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression
from sklearn.utils import ClassifierTags

from credit_risk.data import CATEGORICAL, NUMERIC

# cb_person_default_on_file is excluded: its coefficient has the wrong sign once loan_grade is in the model.
FEATURES = [f for f in NUMERIC + CATEGORICAL if f != "cb_person_default_on_file"]
SCALING = {"pdo": 20, "odds": 50, "scorecard_points": 600}


class PDScorecard(Scorecard):
    """Scorecard that scikit-learn scorers and threshold tuning recognize as a binary classifier.

    Tags are set directly: ClassifierMixin would replace Scorecard.score, which returns points.
    """

    @property
    def classes_(self):
        return self.estimator_.classes_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        return tags


def build_scorecard():
    """Return an unfitted scorecard: 600 points at good:bad odds of 50:1, 20 points to double the odds.

    Numeric features get monotonic bins, features with information value below 0.02 are dropped
    and missing values get the WoE of their own bin (optbinning's default would set it to 0).
    """
    binning = BinningProcess(
        FEATURES,
        categorical_variables=[f for f in FEATURES if f in CATEGORICAL],
        binning_fit_params={f: {"monotonic_trend": "auto_asc_desc"} for f in FEATURES if f in NUMERIC},
        binning_transform_params={f: {"metric_missing": "empirical"} for f in FEATURES},
        selection_criteria={"iv": {"min": 0.02}},
    )
    return PDScorecard(binning, LogisticRegression(max_iter=1000), scaling_method="pdo_odds", scaling_method_params=SCALING)


def points_table(scorecard):
    """Points per bin of the selected features, from a fitted scorecard."""
    table = scorecard.table(style="detailed")
    return table[table["Count"] > 0][["Variable", "Bin", "Count", "Event rate", "WoE", "Points"]]


def iv_table(scorecard):
    """Information value and number of bins per feature, with the selection decision."""
    summary = scorecard.binning_process_.summary()
    return summary[["name", "dtype", "n_bins", "iv", "selected"]].sort_values("iv", ascending=False)
