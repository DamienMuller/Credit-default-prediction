"""Entry point: page config and navigation.

Run with: uv run streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Credit Risk Lab",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("app_pages/insights.py", title="Data insights", icon=":material/insights:"),
    st.Page("app_pages/models.py", title="Models", icon=":material/model_training:"),
    # st.Page("app_pages/predict.py", title="Predict PD", icon=":material/person_search:"),  # disabled for now
]
st.navigation(pages).run()
