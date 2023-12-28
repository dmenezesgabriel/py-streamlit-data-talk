import json

import pandas as pd
import streamlit as st

from src.application.services.llm import LLMService
from src.infrastructure.llm.langchain.client import LLMClient
from src.infrastructure.llm.langchain.utils import format_question, get_primer

st.set_page_config(layout="wide")

available_models = {
    "ChatGPT-4": {"is_enabled": False, "name": "gpt-4"},
    "ChatGPT-3.5": {"is_enabled": False, "name": "gpt-3.5-turbo"},
    "GPT-3": {"is_enabled": False, "name": "text-davinci-003"},
    "GPT-3.5 Instruct": {
        "is_enabled": False,
        "name": "gpt-3.5-turbo-instruct",
    },
    "Code Llama": {"is_enabled": True, "name": "CodeLlama-34b-Instruct-hf"},
}


datasets_urls_files = open("src/resources/dataset_urls.json")
datasets_urls = json.load(datasets_urls_files)
datasets_urls_files.close()


if "datasets" not in st.session_state:
    datasets = {}
    datasets["taxis"] = pd.read_csv(datasets_urls["taxis"])
    datasets["tips"] = pd.read_csv(datasets_urls["tips"])
    st.session_state["datasets"] = datasets
else:
    datasets = st.session_state["datasets"]

with st.sidebar:
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
        dataset_container = st.empty()
        chosen_dataset = dataset_container.radio(
            "datasets: ", datasets.keys(), index=index_no
        )

    with st.expander(":brain: Choose your model(s): ", expanded=True):
        use_model = {}
        for model_desc, model_properties in available_models.items():
            label = f"{model_desc} ({model_properties['name']})"
            key = f"key_{model_desc}"
            use_model[model_desc] = st.checkbox(
                label, value=model_properties["is_enabled"], key=key
            )


openai_key_col, huggingface_key_col2 = st.columns([1, 1])

with st.container():
    with openai_key_col:
        openai_api_key = st.text_input(":key: OpenAI API Key", type="password")
    with huggingface_key_col2:
        hugging_face_api_key = st.text_input(
            ":hugging_face: HuggingFace API Key", type="password"
        )

with st.container():
    prompt = st.text_area(
        ":eyes: What would you like to visualise?", height=10
    )
    generate_viz_button = st.button("Make Viz")


selected_models = [
    model_name
    for model_name, choose_model in use_model.items()
    if choose_model
]
model_count = len(selected_models)

if generate_viz_button and model_count > 0:
    api_keys_entered = True
    # Check API keys are entered.
    if (
        "ChatGPT-4" in selected_models
        or "ChatGPT-3.5" in selected_models
        or "GPT-3" in selected_models
        or "GPT-3.5 Instruct" in selected_models
    ):
        if not openai_api_key.startswith("sk-"):
            st.error("Please enter a valid OpenAI API key.")
            api_keys_entered = False
    if "Code Llama" in selected_models:
        if not hugging_face_api_key.startswith("hf_"):
            st.error("Please enter a valid HuggingFace API key.")
            api_keys_entered = False
    if api_keys_entered:
        llm_client = LLMClient()
        llm_client.keys = {
            "openai": openai_api_key,
            "huggingface": hugging_face_api_key,
        }
        llm_service = LLMService(llm_client)
        # Place for plots depending on how many models
        plots = st.columns(model_count)
        # Get the primer for this dataset
        primer1, primer2 = get_primer(
            datasets[chosen_dataset], 'datasets["' + chosen_dataset + '"]'
        )
        # Create model, run the request and print the results
        for plot_num, model_type in enumerate(selected_models):
            with plots[plot_num]:
                st.subheader(model_type)
                try:
                    # Format the question
                    question_to_ask = format_question(
                        primer1, primer2, prompt, model_type
                    )
                    # Run the question
                    answer = ""
                    answer = llm_service.get_viz_answer_from_prompt(
                        question_to_ask,
                        available_models[model_type]["name"],
                    )
                    answer = primer2 + answer
                    print("Model: " + model_type)
                    print(answer)
                    plot_area = st.empty()
                    plot_area.pyplot(exec(answer))
                except Exception as e:
                    print(e)

tab_list = st.tabs(datasets.keys())
for dataset_num, tab in enumerate(tab_list):
    with tab:
        dataset_name = list(datasets.keys())[dataset_num]
        st.subheader(dataset_name)
        st.dataframe(datasets[dataset_name], hide_index=True)

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
