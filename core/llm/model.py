from langchain_ollama.llms import OllamaLLM
from langchain_openai.chat_models.base import ChatOpenAI
from omegaconf import OmegaConf


class LocalLLM:
    def __init__(self, llm_conf: OmegaConf) -> None:
        self.model = OllamaLLM(**llm_conf)

    def invoke(self, query: str) -> str:
        return self.model.invoke(query)


class OpenAILLM:
    def __init__(self, llm_conf: OmegaConf) -> None:
        self.model = ChatOpenAI(**llm_conf)

    def invoke(self, query: str) -> str:
        return self.model.invoke(query)
