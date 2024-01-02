import ast
import re

import pandas as pd
import streamlit as st


def extract_spec_from_string(string: str) -> str:
    result = "{}"
    pattern = r"st\.vega_lite_chart\(\s*df,\s*({.*?}),\s*(?:use_container_width=(True|False),\s*)?\)"
    match = re.search(pattern, string, re.DOTALL)
    if match:
        result = match.group(1)
    return result


def render_plot_from_model_response(
    model_response: str, dataframe: pd.DataFrame
):
    st.write("Plot: ")
    chart_spec = extract_spec_from_string(model_response)
    if chart_spec:
        with st.expander(label="Vega Spec"):
            st.code(chart_spec, language="json")

        st.vega_lite_chart(
            dataframe,
            ast.literal_eval(chart_spec),
        )
    else:
        st.warning("Vega spec not found in the input string.")
