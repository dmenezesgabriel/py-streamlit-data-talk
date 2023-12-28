class LLMService:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def get_viz_answer_from_prompt(self, question_to_ask, model_type):
        return self.llm_client.get_viz_answer_from_prompt(
            question_to_ask, model_type
        )

    def get_text_answer_from_prompt(self, question_to_ask, model_type):
        return self.llm_client.get_text_answer_from_prompt(
            question_to_ask, model_type
        )
