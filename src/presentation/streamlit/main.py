import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from application.services.llm import LLMService
from infrastructure.llm.langchain.client import LLMClient
from infrastructure.llm.langchain.prompt import (
    dataset_description_by_dtypes,
    generate_viz_prompt,
    viz_code_prompt_template,
)
from presentation.streamlit.constants import available_models
from presentation.streamlit.session import (
    setup_session_datasets,
    setup_session_messages,
)
from presentation.streamlit.utils.logger import configure_st_logger
from presentation.streamlit.utils.vega_lite import (
    render_plot_from_model_response,
)
from utils.resources import ResourceLoader


def set_and_handle_api_key(llm_client, provider, api_key):
    try:
        llm_client.set_api_key(provider, api_key)
    except ValueError as error:
        st.error(error)


def render_assistant_answer(
    selected_models,
    question_to_ask,
    llm_service,
    code_to_execute,
    chosen_dataset,
):
    plots = st.columns(len(selected_models))
    for plot_num, model_type in enumerate(selected_models):
        with plots[plot_num]:
            st.write(model_type)
            try:
                with st.expander("Generated Prompt:"):
                    st.code(question_to_ask, language="markdown")
                answer = llm_service.ask_question(
                    question_to_ask,
                    available_models[model_type]["name"],
                )
                answer = code_to_execute + answer
                with st.expander("Answer"):
                    st.code(answer, language="raw")
                with st.container(border=True):
                    render_plot_from_model_response(
                        answer,
                        st.session_state.datasets[chosen_dataset],
                    )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "chosen_dataset": chosen_dataset,
                    }
                )
            except Exception as e:
                st.error(e)


def main():
    load_dotenv()

    st.set_page_config(layout="wide")

    logger = configure_st_logger()

    resource_loader = ResourceLoader()
    datasets_urls = resource_loader.load_json_file("dataset_urls.json")

    logger.info("Visited main page")

    setup_session_messages()
    setup_session_datasets(datasets_urls)

    llm_client = LLMClient()
    llm_service = LLMService(llm_client)
    llm_client.load_api_keys_from_environment()

    st.header(":eyes: Viz your question", divider="rainbow")

    with st.sidebar:
        with st.expander("Api Keys", expanded=True):
            openapi_api_key = st.text_input(
                ":key: OpenAI API Key",
                type="password",
            )
            huggingface_api_key = st.text_input(
                ":hugging_face: HuggingFace API Key",
                type="password",
            )
        with st.expander(":computer: Upload a csv file (optional)"):
            index_no = 0
            try:
                uploaded_file = st.file_uploader(
                    "Upload a csv file: ", type="csv"
                )
                if uploaded_file:
                    file_name = uploaded_file.name[:-4].capitalize()
                    st.session_state.datasets[file_name] = pd.read_csv(
                        uploaded_file
                    )
                    index_no = len(st.session_state.datasets) - 1
            except Exception as error:
                st.error("Failed to load file, please upload a valid csv")
                print(f"File failed to load. {error}")

        with st.expander(":bar_chart: Choose a dataset", expanded=True):
            chosen_dataset = st.radio(
                "datasets: ", st.session_state.datasets.keys(), index=index_no
            )

        with st.expander(":brain: Choose your model(s): ", expanded=True):
            use_model = {}
            for model_desc, model_properties in available_models.items():
                label = f"{model_desc} ({model_properties['name']})"
                key = f"key_{model_desc}"
                use_model[model_desc] = st.checkbox(
                    label, value=model_properties["default_enabled"], key=key
                )

    if huggingface_api_key:
        set_and_handle_api_key(llm_client, "huggingface", huggingface_api_key)

    if openapi_api_key:
        set_and_handle_api_key(llm_client, "openapi", openapi_api_key)

    with st.expander("Datasets"):
        tab_list = st.tabs(st.session_state.datasets.keys())
        for dataset_num, tab in enumerate(tab_list):
            with tab:
                dataset_name = list(st.session_state.datasets.keys())[
                    dataset_num
                ]
                st.subheader(dataset_name)
                st.dataframe(
                    st.session_state.datasets[dataset_name], hide_index=True
                )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            if message["role"] == "assistant":
                render_plot_from_model_response(
                    message["content"],
                    st.session_state.datasets[message["chosen_dataset"]],
                )

    if prompt := st.chat_input("What would you like to visualize?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            dataset_description = dataset_description_by_dtypes(
                st.session_state.datasets[chosen_dataset]
            )
            code_to_execute = viz_code_prompt_template()
            question_to_ask = generate_viz_prompt(
                dataset_description,
                code_to_execute,
                prompt,
            )
            selected_models = [
                model_name
                for model_name, choose_model in use_model.items()
                if choose_model
            ]
            render_assistant_answer(
                selected_models,
                question_to_ask,
                llm_service,
                code_to_execute,
                chosen_dataset,
            )


#################################################################

# from langchain.agents import create_sql_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# from langchain.chains import LLMChain
# from langchain.llms import HuggingFaceHub, OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.sql_database import SQLDatabase


# database_uri = "duckdb:///../../../database.db"
# database_uri2 = "../../../database.db"
# conn = duckdb.connect(database=database_uri2, read_only=False)

# db = SQLDatabase.from_uri(database_uri)

# dataset_url = st.text_input(
#     "Dataset Url",
#     value="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/raw/taxis.csv",
# )


# if dataset_url:
#     df = pd.read_csv(dataset_url)
#     with st.expander("dataframe"):
#         st.dataframe(df)

#     conn.execute("CREATE TABLE IF NOT EXISTS dataset AS select * from df")


# toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

# agent_executor = create_sql_agent(
#     llm=OpenAI(temperature=0),
#     toolkit=toolkit,
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )

# prompt_template = PromptTemplate.from_template(
#     """You are a data analyst do the below query:
#     {query}.
#     """
# )

# try:
#     st.write(
#         agent_executor.run(
#             prompt_template.format(
#                 query="Describe dataset table",
#             )
#         )
#     )
# except Exception as error:
#     st.warning(error)
