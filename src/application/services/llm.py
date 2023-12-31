class LLMService:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def ask_question(self, question_to_ask: str, model_type: str) -> str:
        return self.llm_client.ask_question(question_to_ask, model_type)
