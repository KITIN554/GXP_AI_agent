from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from omegaconf import OmegaConf


class LocalEmbeddings:
    def __init__(self, embeddings_conf: OmegaConf) -> None:
        self.model = OllamaEmbeddings(**embeddings_conf)

    def embed_query(self, query: str) -> List[float]:
        """_summary_

        Args:
            query (str): _description_

        Returns:
            List[float]: _description_
        """
        return self.model.embed_query(query)

    def get_size_of_embeddings(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """

        return len(self.model.embed_query("Привет"))


class GPTEmbeddings:
    def __init__(self, embeddings_conf: OmegaConf) -> None:
        self.conf = embeddings_conf
        self.model = OpenAIEmbeddings(**embeddings_conf)

    def embed_query(self, query: str) -> List[float]:
        return self.model.embed_query(query)

    def get_size_of_embeddings(self) -> int:
        return self.conf.dimensions
