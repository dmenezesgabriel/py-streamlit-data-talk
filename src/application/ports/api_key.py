from abc import ABC, abstractmethod


class ApiKeyValidator(ABC):
    @abstractmethod
    def is_valid(self, api_key: str) -> bool:
        pass
