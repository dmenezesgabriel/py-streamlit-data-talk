chat_gpt_models = {
    "ChatGPT-4": {"default_enabled": False, "name": "gpt-4"},
    "ChatGPT-3.5": {"default_enabled": False, "name": "gpt-3.5-turbo"},
}
hugging_face_models = {
    "Code Llama": {
        "default_enabled": True,
        "name": "CodeLlama-34b-Instruct-hf",
    },
}
available_models = {**chat_gpt_models, **hugging_face_models}
