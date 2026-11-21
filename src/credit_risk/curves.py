"""Test-set curve data (calibration, ROC, precision-recall) for every fitted model, in long format."""

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def calibration(fitted_models, X, y, n_bins=10):
    """Observed default rate vs mean predicted PD per quantile bin of predicted PD."""
    frames = []
    for (name, remedy), fitted in fitted_models.items():
        observed, predicted = calibration_curve(y, fitted.predict_proba(X)[:, 1], n_bins=n_bins, strategy="quantile")
        frames.append(pd.DataFrame({"model": name, "remedy": remedy, "predicted_pd": predicted, "observed_rate": observed}))
    return pd.concat(frames, ignore_index=True)


def roc(fitted_models, X, y):
    """False and true positive rates over all thresholds."""
    frames = []
    for (name, remedy), fitted in fitted_models.items():
        fpr, tpr, _ = roc_curve(y, fitted.predict_proba(X)[:, 1], drop_intermediate=True)
        frames.append(pd.DataFrame({"model": name, "remedy": remedy, "fpr": fpr, "tpr": tpr}))
    return pd.concat(frames, ignore_index=True)


def precision_recall(fitted_models, X, y):
    """Precision and recall over all thresholds."""
    frames = []
    for (name, remedy), fitted in fitted_models.items():
        precision, recall, _ = precision_recall_curve(y, fitted.predict_proba(X)[:, 1], drop_intermediate=True)
        frames.append(pd.DataFrame({"model": name, "remedy": remedy, "precision": precision, "recall": recall}))
    return pd.concat(frames, ignore_index=True)
