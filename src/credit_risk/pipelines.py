"""Preprocessing, models and class-imbalance remedies assembled into imblearn pipelines."""

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from credit_risk.data import CATEGORICAL, NUMERIC

MODELS = ["Logistic Regression", "LDA", "Decision Tree"]
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
    # Depth and leaf size chosen by 5-fold CV grid search on the training set (average precision).
    return DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=25, random_state=1, class_weight="balanced" if balanced else None
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
    if remedy == "smote":
        return [("smote", SMOTE(random_state=1))]
    return []


def build_pipeline(name, remedy):
    """Assemble preprocessing, optional resampling and the model for one model/remedy combination."""
    steps = [("prep", preprocessor(transform=remedy == "transform"))]
    steps += sampler(remedy)
    steps.append(("model", model(name, balanced=remedy == "class_weight")))
    return Pipeline(steps)
