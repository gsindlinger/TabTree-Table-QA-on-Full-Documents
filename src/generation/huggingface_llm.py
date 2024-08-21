from ..config import Config


class HuggingFaceLLM:
    def __init__(self) -> None:
        self.model = Config.huggingface.model
