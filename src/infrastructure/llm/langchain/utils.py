import pandas as pd


def format_response(res):
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res


def format_question(primer_desc, primer_code, question, model_type):
    # Fill in the model_specific_instructions variable
    instructions = ""
    if model_type == "Code Llama":
        instructions = (
            "Do not use the 'c' argument in the plot function, use 'color' "
            + "instead and only pass color names like 'green', 'red', 'blue'."
        )
    primer_desc = primer_desc.format(instructions)
    return '"""\n' + primer_desc + question + '\n"""\n' + primer_code


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
            primer_desc = (
                primer_desc + "\nThe column '" + i + "' has categorical values"
            )
        elif (
            df_dataset.dtypes[i] == "int64"
            or df_dataset.dtypes[i] == "float64"
        ):
            primer_desc = (
                primer_desc
                + "\nThe column '"
                + i
                + "' is type "
                + str(df_dataset.dtypes[i])
            )
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."

    primer_desc = (
        primer_desc + "{}"
    )  # Space for additional instructions if needed
    primer_desc = (
        primer_desc
        + "\nUsing Python version 3.9.12, create a script using the "
        + "dataframe df to graph the following: "
    )
    return primer_desc


def make_viz_code(df_dataset, df_name):
    primer_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    primer_code = primer_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    primer_code = (
        primer_code
        + "ax.spines['top'].set_visible(False)\nax.spines['right']"
        + ".set_visible(False) \n"
    )
    primer_code = primer_code + "df=" + df_name + ".copy()\n"
    return primer_code


def get_primer(df_dataset, df_name):
    # Primer function to take a dataframe and its name
    # and the name of the columns
    # and any columns with less than 20 unique values it adds the values
    # to the primer and horizontal grid lines and labeling
    primer_desc = make_dataset_description(df_dataset)
    primer_code = make_viz_code(df_dataset, df_name)
    return primer_desc, primer_code
