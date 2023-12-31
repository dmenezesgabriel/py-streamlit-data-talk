from textwrap import dedent

import pandas as pd


def generate_viz_prompt(
    df_description: str, viz_code: str, question: str
) -> str:
    instructions = dedent(
        """
        Generate chart configuration for the given dataset based on a user
        question.
        User question:
        """
    )
    return dedent(
        f"""
        {df_description}
        {instructions} {question}
        {viz_code}
        """
    )


def dataset_description_by_dtypes(df_dataset: pd.DataFrame) -> str:
    return dedent(
        f"""
        Use a dataframe called df from data_file.csv. This is the result of `print(df.dtypes)`:

        {str(df_dataset.dtypes.to_markdown())}
        """
    )


def viz_code_prompt_template():
    return dedent(
        """
        import streamlit as st
        import pandas as pd
        st.vega_lite_chart(
        """
    )
