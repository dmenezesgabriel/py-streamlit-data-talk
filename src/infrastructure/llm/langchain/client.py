import openai
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

from src.infrastructure.llm.langchain.utils import format_response


class LLMClient:
    def __init__(self, keys=None):
        self._keys = keys

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, value):
        self._keys = value

    def get_viz_answer_from_prompt(self, question_to_ask, model_type):
        if model_type == "gpt-4" or model_type == "gpt-3.5-turbo":
            # Run OpenAI ChatCompletion API
            task = "Generate Python Code Script."
            if model_type == "gpt-4":
                # Ensure GPT-4 does not include additional comments
                task = (
                    task + " The script should only include code, no comments."
                )
            openai.api_key = self.keys["openai"]
            response = openai.ChatCompletion.create(
                model=model_type,
                messages=[
                    {"role": "system", "content": task},
                    {"role": "user", "content": question_to_ask},
                ],
            )
            llm_response = response["choices"][0]["message"]["content"]
        elif (
            model_type == "text-davinci-003"
            or model_type == "gpt-3.5-turbo-instruct"
        ):
            # Run OpenAI Completion API
            openai.api_key = self.keys["openai"]
            response = openai.Completion.create(
                engine=model_type,
                prompt=question_to_ask,
                temperature=0,
                max_tokens=500,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["plt.show()"],
            )
            llm_response = response["choices"][0]["text"]
        else:
            # Hugging Face model
            llm = HuggingFaceHub(
                huggingfacehub_api_token=self.keys["huggingface"],
                repo_id="codellama/" + model_type,
                model_kwargs={"temperature": 0.1, "max_new_tokens": 500},
            )
            llm_prompt = PromptTemplate.from_template(question_to_ask)
            llm_chain = LLMChain(llm=llm, prompt=llm_prompt)
            llm_response = llm_chain.predict()
        # rejig the response
        llm_response = format_response(llm_response)
        return llm_response

    def get_text_answer_from_prompt(self, question_to_ask, model_type):
        pass
