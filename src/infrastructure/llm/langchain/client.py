import os

from application.ports.llm import LanguageModel
from infrastructure.llm.langchain.llm import GPT4, CodeLlama, GPT35Turbo


class LanguageModelFactory:
    @staticmethod
    def create_model(model_type: str, api_key: str) -> LanguageModel:
        if model_type == "gpt-4":
            return GPT4(api_key, model_type)
        elif model_type == "gpt-3.5-turbo":
            return GPT35Turbo(api_key, model_type)
        else:
            return CodeLlama(api_key, model_type)


class LLMClient:
    def __init__(self, keys=None):
        self._keys = keys

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, value):
        self._keys = value

    def load_api_keys_from_environment(self):
        self._keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
        }

    def ask_question(self, question_to_ask, model_type):
        key_map = {
            "gpt-4": "openapi",
            "gpt-3.5-turbo": "openapi",
            "CodeLlama-34b-Instruct-hf": "huggingface",
        }
        model = LanguageModelFactory.create_model(
            model_type, self.keys.get(key_map[model_type], "")
        )
        return model.ask_question(question_to_ask)
