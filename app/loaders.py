"""Cached loaders for the dataset, saved results and PD models."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from credit_risk.data import load_loans, split

ARTIFACTS = Path("artifacts")


@st.cache_data
def loans():
    """Cleaned loan dataset."""
    return load_loans()


@st.cache_data
def artifact(name):
    """A results table saved by credit_risk.train, e.g. 'test_results.csv'."""
    return pd.read_csv(ARTIFACTS / name)


@st.cache_resource
def pd_model(name):
    """A fitted PD model: 'boosting', 'scorecard' or 'logistic'."""
    return joblib.load(ARTIFACTS / f"pd_{name}.joblib")


@st.cache_data
def test_default_rate():
    """Actual default rate of the test set used for all results."""
    _, _, _, y_test = split(load_loans())
    return y_test.mean()
