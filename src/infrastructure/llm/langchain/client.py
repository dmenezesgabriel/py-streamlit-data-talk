import os

from application.ports.api_key import ApiKeyValidator
from application.ports.llm import LanguageModel
from infrastructure.llm.langchain.llm import GPT4, CodeLlama, GPT35Turbo


class HuggingFaceApiKeyValidator(ApiKeyValidator):
    def is_valid(self, api_key: str) -> bool:
        return api_key.startswith("hf_")


class OpenAIApiKeyValidator(ApiKeyValidator):
    def is_valid(self, api_key: str) -> bool:
        return api_key.startswith("sk-")


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
    def __init__(self, keys=None, api_key_validators=None):
        self._keys = keys or {}
        self._api_key_validators = api_key_validators or {
            "openai": OpenAIApiKeyValidator(),
            "huggingface": HuggingFaceApiKeyValidator(),
        }

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, value):
        for provider, api_key in value.items():
            if not self.validate_api_key(api_key, provider):
                raise ValueError(f"Invalid API key for {provider} model.")
        self._keys = value

    def set_api_key(self, provider: str, api_key: str):
        if self.validate_api_key(api_key, provider):
            self._keys[provider] = api_key
        else:
            raise ValueError(f"Invalid API key for {provider} model.")

    def load_api_keys_from_environment(self):
        api_key_mappings = {
            "openai": "OPENAI_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }

        for provider, env_var in api_key_mappings.items():
            api_key = os.getenv(env_var)
            if api_key:
                self.set_api_key(provider, api_key)

    def validate_api_key(self, api_key: str, provider: str) -> bool:
        if provider in self._api_key_validators:
            return self._api_key_validators[provider].is_valid(api_key)
        else:
            return False

    def ask_question(self, question_to_ask, model_type):
        key_map = {
            "gpt-4": "openai",
            "gpt-3.5-turbo": "openai",
            "CodeLlama-34b-Instruct-hf": "huggingface",
        }

        provider = key_map.get(model_type)
        api_key = self.keys.get(provider, "")

        model = LanguageModelFactory.create_model(model_type, api_key)
        return model.ask_question(question_to_ask)
