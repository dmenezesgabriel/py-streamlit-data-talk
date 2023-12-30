def hugging_face_api_key_is_valid(api_key: str):
    return api_key.startswith("hf_")


def openai_api_key_is_valid(api_key: str):
    return api_key.startswith("sk-")
