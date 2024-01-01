import ast
import os
import random
import re
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.application.services.llm import LLMService
from src.infrastructure.llm.langchain.client import LLMClient
from src.infrastructure.llm.langchain.utils import (
    dataset_description_by_dtypes,
    format_question,
    make_viz_code,
)
from src.infrastructure.llm.utils.api_key import (
    hugging_face_api_key_is_valid,
    openai_api_key_is_valid,
)
from src.presentation.streamlit.constants import available_models
from src.presentation.streamlit.session import (
    setup_session_datasets,
    setup_session_messages,
)
from src.presentation.streamlit.utils.logger import configure_st_logger
from src.utils.resources import ResourceLoader

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

st.set_page_config(layout="wide")

logger = configure_st_logger()

resource_loader = ResourceLoader()
datasets_urls = resource_loader.load_json_file("dataset_urls.json")

logger.info("Started")

setup_session_messages()
datasets = setup_session_datasets(datasets_urls)

st.title(":eyes: Viz your question")
with st.sidebar:
    openai_api_key = st.text_input(":key: OpenAI API Key", type="password")
    hugging_face_api_key = st.text_input(
        ":hugging_face: HuggingFace API Key", type="password"
    )
    with st.expander(":computer: Upload a csv file (optional)"):
        index_no = 0
        try:
            uploaded_file = st.file_uploader("Upload a csv file: ", type="csv")
            if uploaded_file:
                file_name = uploaded_file.name[:-4].capitalize()
                datasets[file_name] = pd.read_csv(uploaded_file)
                index_no = len(datasets) - 1
        except Exception as error:
            st.error("Failed to load file, please upload a valid csv")
            print(f"File failed to load. {error}")

    with st.expander(":bar_chart: Choose a dataset", expanded=True):
        chosen_dataset = st.radio(
            "datasets: ", datasets.keys(), index=index_no
        )

    with st.expander(":brain: Choose your model(s): ", expanded=True):
        use_model = {}
        for model_desc, model_properties in available_models.items():
            label = f"{model_desc} ({model_properties['name']})"
            key = f"key_{model_desc}"
            use_model[model_desc] = st.checkbox(
                label, value=model_properties["default_enabled"], key=key
            )

selected_models = [
    model_name
    for model_name, choose_model in use_model.items()
    if choose_model
]
selected_model_count = len(selected_models)


with st.expander("Datasets"):
    tab_list = st.tabs(datasets.keys())
    for dataset_num, tab in enumerate(tab_list):
        with tab:
            dataset_name = list(datasets.keys())[dataset_num]
            st.subheader(dataset_name)
            st.dataframe(datasets[dataset_name], hide_index=True)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What would you like to visualize?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if selected_model_count > 0:
            api_keys_entered = True
            if selected_models in [
                "ChatGPT-4",
                "ChatGPT-3.5",
                "GPT-3",
                "GPT-3.5 Instruct",
            ]:
                if not openai_api_key_is_valid(openai_api_key):
                    st.error("Please enter a valid OpenAI API key.")
                    api_keys_entered = False
            if "Code Llama" in selected_models:
                hugging_face_api_key = (
                    hugging_face_api_key or HUGGINGFACE_API_KEY
                )
                if not hugging_face_api_key_is_valid(hugging_face_api_key):
                    st.error("Please enter a valid HuggingFace API key.")
                    api_keys_entered = False
            if api_keys_entered:
                llm_client = LLMClient()
                llm_client.keys = {
                    "openai": openai_api_key,
                    "huggingface": hugging_face_api_key,
                }
                llm_service = LLMService(llm_client)
                plots = st.columns(selected_model_count)
                expected_description = dataset_description_by_dtypes(
                    datasets[chosen_dataset]
                )
                code_to_execute = make_viz_code(
                    'datasets["' + chosen_dataset + '"]'
                )
                for plot_num, model_type in enumerate(selected_models):
                    with plots[plot_num]:
                        st.write(model_type)
                        try:
                            # Format the question
                            question_to_ask = format_question(
                                expected_description,
                                code_to_execute,
                                prompt,
                            )
                            with st.expander("Generated Prompt:"):
                                st.code(question_to_ask, language="markdown")
                            # Run the question
                            answer = ""
                            answer = llm_service.ask_question(
                                question_to_ask,
                                available_models[model_type]["name"],
                            )
                            answer = code_to_execute + answer
                            with st.expander("Answer"):
                                st.code(answer, language="raw")
                            vega_spec_pattern = (
                                r"st\.vega_lite_chart\(df, ({.*?})\)"
                            )
                            match = re.search(
                                vega_spec_pattern, answer, re.DOTALL
                            )
                            with st.container(border=True):
                                st.write("Plot: ")
                                if match:
                                    vega_spec_dict = match.group(1)
                                    with st.expander(label="Vega Spec"):
                                        st.code(
                                            vega_spec_dict, language="json"
                                        )
                                    st.vega_lite_chart(
                                        datasets[chosen_dataset],
                                        ast.literal_eval(vega_spec_dict),
                                    )
                                else:
                                    st.write(
                                        "Vega spec not found in the input string."
                                    )
                        except Exception as e:
                            st.error(e)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})


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
