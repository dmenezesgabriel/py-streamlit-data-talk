import pandas as pd


def format_question(
    expected_description: str, viz_code: str, question: str
) -> str:
    instructions = "\n\nusing the following query: "
    expected_description = expected_description.format(instructions)
    return '"""\n' + expected_description + question + '\n"""\n\n' + viz_code


def describe_dataframe_columns(dataframe: pd.DataFrame) -> str:
    columns = "'" + "', '".join(str(x) for x in dataframe.columns) + "'"
    return (
        "Use a dataframe called df from data_file.csv with columns: \n\n"
        f"{columns}"
        ". "
        "\n\n"
    )


def make_dataset_description(df_dataset) -> str:
    primer_desc = describe_dataframe_columns(df_dataset)
    for i in df_dataset.columns:
        if df_dataset.dtypes[i] == "O":
            primer_desc += (
                "\n\n- The column '" + i + "' has categorical values"
            )
        elif (
            df_dataset.dtypes[i] == "int64"
            or df_dataset.dtypes[i] == "float64"
        ):
            primer_desc += (
                "\n\n- The column '"
                + i
                + "' is type "
                + str(df_dataset.dtypes[i])
            )
    primer_desc = primer_desc + "{}"
    return primer_desc


def make_viz_code(df_name):
    viz_code = (
        "import streamlit as st\n"
        "import pandas as pd\n"
        f"df={df_name}.copy()\n"
        "st.vega_lite_chart("
    )
    return viz_code
