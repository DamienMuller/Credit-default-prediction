"""Readable names for features and categories."""

import math

FEATURES = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_emp_length": "Employment length",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan-to-income ratio",
    "cb_person_cred_hist_length": "Credit history length",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan intent",
    "loan_grade": "Loan grade",
    "cb_person_default_on_file": "Previous default on file",
}

CATEGORIES = {
    "RENT": "Rent",
    "MORTGAGE": "Mortgage",
    "OWN": "Own",
    "OTHER": "Other",
    "EDUCATION": "Education",
    "MEDICAL": "Medical",
    "VENTURE": "Venture",
    "PERSONAL": "Personal",
    "DEBTCONSOLIDATION": "Debt consolidation",
    "HOMEIMPROVEMENT": "Home improvement",
}


FORMATS = {
    "person_income": "${:,.0f}",
    "loan_amnt": "${:,.0f}",
    "loan_int_rate": "{:.2f}%",
    "loan_percent_income": "{:.2f}",
}
YEARS = {"person_emp_length"}


def bin_text(feature, bin_):
    """Readable bin, e.g. '[49986.00, 59853.50)' -> '$49,986 to $59,854'; categorical and Missing bins via category()."""
    if (feature not in FORMATS and feature not in YEARS) or not bin_.startswith(("(", "[")):
        return category(bin_)
    low, high = (float(edge) for edge in bin_.strip("[()").split(", "))
    if feature in YEARS:
        return years_text(low, high)
    fmt = FORMATS[feature].format
    if low == float("-inf"):
        return f"below {fmt(high)}"
    if high == float("inf"):
        return f"{fmt(low)} or more"
    return f"{fmt(low)} to {fmt(high)}"


def years_text(low, high):
    """Whole-year range of a bin on an integer feature, e.g. (2.5, 4.5) -> '3 to 4 years'."""
    first = math.ceil(low) if math.isfinite(low) else 0
    if not math.isfinite(high):
        return f"{first} years or more"
    last = math.ceil(high) - 1
    if first == last:
        return "1 year" if first == 1 else f"{first} years"
    return f"{first} to {last} years"


def category(value):
    """Readable category names, e.g. 'RENT, OTHER' -> 'Rent, Other'; other values are returned unchanged."""
    return ", ".join(CATEGORIES.get(part, part) for part in value.split(", "))


MODELS = ["Gradient Boosting", "Decision Tree", "Scorecard", "Logistic Regression", "LDA"]

REMEDIES = {
    "baseline": "No remedy",
    "transform": "Yeo-Johnson transform",
    "class_weight": "Class weights",
    "oversample": "Oversampling",
    "undersample": "Downsampling",
    "over_under": "Over- + downsampling",
    "smote": "SMOTE",
}
