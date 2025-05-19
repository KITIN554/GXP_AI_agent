from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM
from langchain_core.vectorstores.base import VectorStore
from omegaconf import OmegaConf


class RAG:
    def __init__(
        self,
        vector_store: VectorStore,
        model: BaseLLM,
        generation_prompt: str,
        search_config: OmegaConf,
    ):
        self.vector_store = vector_store
        self.model = model
        self.generation_prompt = PromptTemplate.from_template(generation_prompt)
        self.search_config = search_config

    def collect_context(self, query: str, return_raw_docs: bool = False) -> str:
        # results = self.vector_store.query_retriever(query, self.search_config)
        results = self.vector_store.query_ensemble(query, self.search_config)
        # results = self.vector_store.query_rerank(query, self.search_config)
        if return_raw_docs:
            return [result.metadata["source"] for result in results]
        return self.vector_store.collect_docs(results)

    def collect_prompt(self, query: str, context: str) -> str:
        return self.generation_prompt.invoke({"question": query, "context": context})

    def invoke(self, query: str) -> str:
        relevant_docs = self.collect_context(query, return_raw_docs=True)
        prompt = self.collect_prompt(query, relevant_docs)

        return {"context": relevant_docs, "answer": self.model.invoke(prompt)}
