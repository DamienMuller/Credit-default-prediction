"""Preprocessing, models and class-imbalance remedies assembled into imblearn pipelines."""

from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from credit_risk.data import CATEGORICAL, NUMERIC
from credit_risk.scorecard import build_scorecard

MODELS = ["Logistic Regression", "LDA", "Decision Tree", "Gradient Boosting"]
REMEDIES = ["baseline", "transform", "class_weight", "oversample", "undersample", "over_under", "smote"]


def preprocessor(transform):
    """Impute and scale numerics (Yeo-Johnson if transform, else standardize) and one-hot encode categoricals."""
    scaler = PowerTransformer(method="yeo-johnson") if transform else StandardScaler()
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="mean"), scaler), NUMERIC),
        ("cat", OneHotEncoder(drop="first"), CATEGORICAL),
    ])


def model(name, balanced=False):
    """Return a fresh classifier; balanced=True weights both classes equally."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced" if balanced else None)
    if name == "LDA":
        return LinearDiscriminantAnalysis(priors=[0.5, 0.5] if balanced else None)
    # Tree and boosting hyperparameters chosen by 5-fold CV grid search on the training set (average precision).
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=25, random_state=1, class_weight="balanced" if balanced else None
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=1.0,
        max_iter=1000, early_stopping=True, random_state=1, class_weight="balanced" if balanced else None,
    )


def sampler(remedy):
    """Return the resampling steps for a remedy (empty for non-resampling remedies)."""
    if remedy == "oversample":
        return [("over", RandomOverSampler(random_state=1))]
    if remedy == "undersample":
        return [("under", RandomUnderSampler(random_state=1))]
    if remedy == "over_under":
        return [
            ("over", RandomOverSampler(sampling_strategy=0.5, random_state=1)),
            ("under", RandomUnderSampler(sampling_strategy=0.6, random_state=1)),
        ]
    return []


def smotenc_steps():
    """Scale numerics, apply SMOTENC on raw categories, then one-hot encode.

    SMOTENC must see categories before encoding: plain SMOTE on one-hot columns
    creates impossible fractional categories (e.g. 0.6 renter, 0.4 owner).
    """
    scale = ColumnTransformer(
        [("num", make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), NUMERIC), ("cat", "passthrough", CATEGORICAL)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    encode = ColumnTransformer([("cat", OneHotEncoder(drop="first"), CATEGORICAL)], remainder="passthrough")
    return [("scale", scale), ("smotenc", SMOTENC(categorical_features=CATEGORICAL, random_state=1)), ("encode", encode)]


def build_pipeline(name, remedy):
    """Assemble preprocessing, optional resampling and the model for one model/remedy combination.

    The scorecard bins raw features itself and is only built without remedy.
    """
    if name == "Scorecard":
        return build_scorecard()
    if remedy == "smote":
        steps = smotenc_steps()
    else:
        steps = [("prep", preprocessor(transform=remedy == "transform"))] + sampler(remedy)
    steps.append(("model", model(name, balanced=remedy == "class_weight")))
    return Pipeline(steps)
