from pathlib import Path
from typing import List
from uuid import uuid4
import dill
from faiss import Index
from langchain_community.docstore.base import Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from loguru import logger
from omegaconf import OmegaConf


class VectorDB:
    def __init__(
        self,
        faiss_dir: Path,
        embedding_fn: Embeddings,
        index: Index,
        docstore: Docstore = InMemoryDocstore(),
        index_to_docstore_id={},
    ) -> None:
        self.faiss_dir = faiss_dir
        self.embedding_fn = embedding_fn
        self.index = index
        self.vector_store = self._create_or_load()

    def _create_or_load(self):
        match any(
            Path(self.faiss_dir).glob("*.faiss")
        ):  # Проверяем есть ли файлы в папке
            case True:
                logger.info(f"Загружаю локальную БД: {self.faiss_dir}")
                vector_store = FAISS.load_local(
                    self.faiss_dir,
                    self.embedding_fn,
                    allow_dangerous_deserialization=True,
                )
            case False:
                logger.info(f"Создаю локальную БД: {self.faiss_dir}")
                vector_store = FAISS(
                    embedding_function=self.embedding_fn,
                    index=self.index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
                logger.success(f"Сохранил БД: {self.faiss_dir}")
            case _:
                raise ValueError("Ошибка чтения/создания БД!")
        return vector_store

    def add_documents(self, documents: List[Document]) -> List[str]:
        uuids = [[str(uuid4()) for _ in range(len(doc_list))] for doc_list in documents]
        for idx, (uuid, document) in enumerate(zip(uuids, documents), start=1):
            logger.debug(
                f"{idx} / {len(documents)} - {self.vector_store.add_documents(document, ids=uuid)}"
            )
        self.save_db()
        return True

    def query_directly(
        self, query: str, search_config_directly: OmegaConf
    ) -> List[Document]:
        return self.vector_store.similarity_search(query, **search_config_directly)

    def query_retriever(
        self, query: str, search_config_retriever: OmegaConf
    ) -> List[Document]:
        retriever = self.vector_store.as_retriever(**search_config_retriever)
        return retriever.invoke(query)

    def create_bm25_retriever(self):
        with open(Path(self.faiss_dir).parent / "docs.dill", "rb") as file:
            docs = dill.load(file)
        bm25_retriever = BM25Retriever.from_documents(sum(docs, []))
        bm25_retriever.k = 3
        return bm25_retriever

    def create_flash_rerank(self, base_retriever):
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        return compression_retriever

    def query_ensemble(self, query: str, search_config_retriever) -> List[Document]:
        bm25_retriever = self.create_bm25_retriever()
        faiss_retriever = self.vector_store.as_retriever(**search_config_retriever)
        flash_retriever = self.create_flash_rerank(faiss_retriever)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever, flash_retriever],
            weights=[0.7, 0.2, 0.1],
        )

        return ensemble_retriever.invoke(query)

    def query_rerank(self, query, search_config_retriever) -> List[Document]:
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vector_store.as_retriever(**search_config_retriever),
        )
        return compression_retriever.invoke(query)

    def collect_docs(self, relevant_docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in relevant_docs)

    def save_db(self):
        self.vector_store.save_local(self.faiss_dir)
