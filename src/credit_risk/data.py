"""Load, clean and split the loan dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "loan_status"
NUMERIC = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def load_loans(path="loan_data.csv"):
    """Read the dataset and drop implausible ages and employment lengths (> 100 years)."""
    loans = pd.read_csv(path)
    implausible = (loans["person_age"] > 100) | (loans["person_emp_length"] > 100)
    return loans[~implausible].reset_index(drop=True)


def split(loans, test_size=0.2, seed=0):
    """Stratified train/test split returning X_train, X_test, y_train, y_test."""
    X = loans[NUMERIC + CATEGORICAL]
    y = loans[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
