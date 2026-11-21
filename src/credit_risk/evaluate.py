"""Cross-validated comparison on the training set and single evaluation on the test set."""

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV, cross_validate

from credit_risk.pipelines import build_pipeline

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def cv_scores(X, y, models, remedies):
    """Return mean and std of ROC AUC, average precision and Brier score for every model/remedy pair."""
    rows = []
    for name in models:
        for remedy in remedies:
            s = cross_validate(
                build_pipeline(name, remedy), X, y, cv=CV,
                scoring=["roc_auc", "average_precision", "neg_brier_score"], n_jobs=-1,
            )
            rows.append({
                "model": name,
                "remedy": remedy,
                "roc_auc": s["test_roc_auc"].mean(),
                "roc_auc_std": s["test_roc_auc"].std(),
                "ap": s["test_average_precision"].mean(),
                "ap_std": s["test_average_precision"].std(),
                "brier": -s["test_neg_brier_score"].mean(),
            })
    return pd.DataFrame(rows)


def tuned_model(name, remedy):
    """Wrap a pipeline so its F1-maximising threshold is chosen by CV on the training data."""
    return TunedThresholdClassifierCV(build_pipeline(name, remedy), scoring="f1", cv=CV)


def ks_statistic(y, prob):
    """Kolmogorov-Smirnov statistic: maximum gap between the score distributions of defaults and non-defaults."""
    fpr, tpr, _ = roc_curve(y, prob)
    return (tpr - fpr).max()


def test_metrics(fitted, X, y):
    """Score a fitted TunedThresholdClassifierCV on held-out data."""
    prob = fitted.predict_proba(X)[:, 1]
    pred = fitted.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    auc = roc_auc_score(y, prob)
    return {
        "threshold": fitted.best_threshold_,
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, prob),
        "ap": average_precision_score(y, prob),
        "brier": brier_score_loss(y, prob),
        "mean_pd": prob.mean(),
        "f1_at_0.5": f1_score(y, fitted.estimator_.predict(X)),
        "f1": f1_score(y, pred),
        "sensitivity": tp / (tp + fn),
        "precision": tp / (tp + fp),
        "specificity": tn / (tn + fp),
    }


def fit_tuned_models(X_train, y_train, models, remedies):
    """Fit every threshold-tuned model/remedy pair on the training set, keyed by (model, remedy)."""
    return {(name, remedy): tuned_model(name, remedy).fit(X_train, y_train) for name in models for remedy in remedies}


def test_scores(fitted_models, X_test, y_test):
    """Score every fitted model once on the test set."""
    rows = [
        {"model": name, "remedy": remedy, **test_metrics(fitted, X_test, y_test)}
        for (name, remedy), fitted in fitted_models.items()
    ]
    return pd.DataFrame(rows)
