import openai
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

from application.ports.llm import LanguageModel


class GPT4(LanguageModel):
    def __init__(self, api_key: str, model_type: str):
        self.api_key = api_key
        self.model_type = model_type

    def ask_question(self, question_to_ask: str) -> str:
        task = (
            "Generate Python Code Script. "
            "The script should only include code, no comments."
        )
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model_type,
            messages=[
                {"role": "system", "content": task},
                {"role": "user", "content": question_to_ask},
            ],
        )
        return response["choices"][0]["message"]["content"]


class GPT35Turbo(LanguageModel):
    def __init__(self, api_key: str, model_type: str):
        self.api_key = api_key
        self.model_type = model_type

    def ask_question(self, question_to_ask: str) -> str:
        openai.api_key = self.api_key
        response = openai.Completion.create(
            engine=self.model_type,
            prompt=question_to_ask,
            temperature=0,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response["choices"][0]["text"]


class CodeLlama(LanguageModel):
    def __init__(self, api_key: str, model_type: str):
        self.api_key = api_key
        self.model_type = model_type

    def ask_question(self, question_to_ask: str) -> str:
        llm = HuggingFaceHub(
            huggingfacehub_api_token=self.api_key,
            repo_id="codellama/" + self.model_type,
            model_kwargs={"max_new_tokens": 500},
        )
        llm_prompt = PromptTemplate.from_template(question_to_ask)
        llm_chain = LLMChain(llm=llm, prompt=llm_prompt, verbose=True)
        return llm_chain.predict()
