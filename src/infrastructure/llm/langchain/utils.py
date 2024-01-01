from textwrap import dedent

import pandas as pd


def format_question(df_description: str, viz_code: str, question: str) -> str:
    instructions = "\n\nGenerate a chart with the following query: "
    return (
        '"""\n'
        + df_description
        + instructions
        + question
        + '\n"""\n\n'
        + viz_code
    )


def dataset_description_by_dtypes(df_dataset: pd.DataFrame) -> str:
    return dedent(
        f"""
        Use a dataframe called df from data_file.csv.
        This is the result of `print(df.dtypes)`:

        {str(df_dataset.dtypes.to_markdown())}
        """
    )


def make_viz_code(df_name):
    viz_code = (
        "import streamlit as st\n"
        "import pandas as pd\n"
        f"df={df_name}.copy()\n"
        "st.vega_lite_chart("
    )
    return viz_code
