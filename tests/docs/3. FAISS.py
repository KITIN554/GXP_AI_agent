import sys

sys.path.append("../")
sys.path.append("./")

from langchain_core.documents.base import Document
from loguru import logger
from omegaconf import OmegaConf

from core.embeddings.model import LocalEmbeddings
from core.vectordb.index import Index
from core.vectordb.vectore_store import VectorDB

if __name__ == "__main__":
    conf = OmegaConf.load("./config/config.yaml")

    if not conf.logging.debug:
        logger.remove()

    model = LocalEmbeddings(conf.embeddings.ollama_config)
    index = Index(model.get_size_of_embeddings())
    store = VectorDB(conf.faiss_config.store.data_path, model.model, index.index)

    logger.debug(store.add_documents([Document("asjk")]))

    logger.debug(
        store.query_retriever("рудд", conf.faiss_config.search_config.retriever)
    )
