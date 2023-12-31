import json
import os


class ResourceLoader:
    def __init__(self, base_path: str = "src/resources"):
        self.base_path = base_path

    def load_text_file(self, file_name: str) -> str:
        absolute_path = os.path.join(self.base_path, file_name)
        with open(absolute_path, "r") as f:
            content = f.read()
        return content

    def load_json_file(self, file_name: str) -> dict:
        absolute_path = os.path.join(self.base_path, file_name)
        with open(absolute_path, "r") as f:
            content = json.load(f)
        return content
