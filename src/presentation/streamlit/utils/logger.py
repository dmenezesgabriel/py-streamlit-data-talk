import streamlit as st

from src.infrastructure.logger.logger import configure_logger


@st.cache_resource
def configure_st_logger():
    return configure_logger()
