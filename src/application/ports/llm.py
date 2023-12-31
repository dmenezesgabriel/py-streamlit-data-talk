from abc import ABC, abstractmethod


class LanguageModel(ABC):
    @abstractmethod
    def ask_question(self, question_to_ask: str) -> str:
        pass
