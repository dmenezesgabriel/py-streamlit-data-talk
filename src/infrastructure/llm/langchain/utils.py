import pandas as pd


def format_question(expected_description, viz_code, question, model_type):
    # Fill in the model_specific_instructions variable
    instructions = ""
    if model_type == "Code Llama":
        instructions = (
            "Do not use the 'c' argument in the plot function, use 'color' "
            + "instead and only pass color names like 'green', 'red', 'blue'."
        )
    expected_description = expected_description.format(instructions)
    return '"""\n' + expected_description + question + '\n"""\n' + viz_code


def describe_dataframe_columns(dataframe: pd.DataFrame):
    return (
        "Use a dataframe called df from data_file.csv with columns '"
        + "','".join(str(x) for x in dataframe.columns)
        + "'. "
    )


def make_dataset_description(df_dataset):
    primer_desc = describe_dataframe_columns(df_dataset)
    for i in df_dataset.columns:
        if df_dataset.dtypes[i] == "O":
            primer_desc += "\nThe column '" + i + "' has categorical values"
        elif (
            df_dataset.dtypes[i] == "int64"
            or df_dataset.dtypes[i] == "float64"
        ):
            primer_desc += (
                "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i])
            )
    primer_desc = (
        primer_desc
        + "\nLabel the x and y axes appropriately."
        + "{}"
        + "\nUsing Python version 3.9.12, create a script using the "
        + "dataframe df to graph the following: "
    )
    return primer_desc


def make_viz_code(df_dataset, df_name):
    viz_code = (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
        "ax.spines['top'].set_visible(False)\n"
        "ax.spines['right'].set_visible(False) \n"
        f"df={df_name}.copy()\n"
    )
    return viz_code
