import pandas as pd
import streamlit as st


def setup_session_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def setup_session_datasets(datasets_urls: dict):
    if "datasets" not in st.session_state:
        datasets_names = datasets_urls.keys()
        datasets = {
            name: pd.read_csv(url)
            for name, url in datasets_urls.items()
            if name in datasets_names
        }
        st.session_state.datasets = datasets
    else:
        datasets = st.session_state["datasets"]
    return datasets
