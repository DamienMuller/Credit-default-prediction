"""Run the full comparison and save results, curves and the PD models used by the app.

Usage: uv run python -m credit_risk.train
"""

from pathlib import Path

import joblib

from credit_risk import curves
from credit_risk.data import load_loans, split
from credit_risk.evaluate import cv_scores, fit_tuned_models, test_scores
from credit_risk.pipelines import MODELS, REMEDIES

ARTIFACTS = Path("artifacts")
PD_MODELS = {"pd_logistic.joblib": ("Logistic Regression", "baseline"), "pd_boosting.joblib": ("Gradient Boosting", "baseline")}


def save_csv(df, name):
    """Write a DataFrame to the artifacts folder with 6 decimals."""
    df.round(6).to_csv(ARTIFACTS / name, index=False)


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = split(load_loans())

    cv = cv_scores(X_train, y_train, MODELS, REMEDIES)
    fitted = fit_tuned_models(X_train, y_train, MODELS, REMEDIES)
    test = test_scores(fitted, X_test, y_test)

    save_csv(cv, "cv_results.csv")
    save_csv(test, "test_results.csv")
    save_csv(curves.calibration(fitted, X_test, y_test), "calibration.csv")
    save_csv(curves.roc(fitted, X_test, y_test), "roc.csv")
    save_csv(curves.precision_recall(fitted, X_test, y_test), "precision_recall.csv")
    for file, key in PD_MODELS.items():
        joblib.dump(fitted[key], ARTIFACTS / file)

    print("5-fold CV on training set\n")
    print(cv.round(3).to_string(index=False))
    print("\nTest set (thresholds tuned by CV on training set)\n")
    print(test.round(3).to_string(index=False))
    print(f"\nArtifacts saved to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
