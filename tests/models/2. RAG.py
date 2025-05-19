import sys

sys.path.append("../")
sys.path.append("./")

from loguru import logger
from omegaconf import OmegaConf

from core.embeddings.model import LocalEmbeddings
from core.llm.model import LocalLLM
from core.vectordb.index import Index
from core.vectordb.vectore_store import VectorDB
from core.llm.rag_pipeline import RAG
from core.prompts.llm_prompt import rag_prompt

if __name__ == "__main__":
    conf = OmegaConf.load("./config/config.yaml")
    if not conf.logging.debug:
        logger.remove()
    llm = LocalLLM(conf.llm.ollama_config)
    embedddings = LocalEmbeddings(conf.embeddings.ollama_config)
    index = Index(embedddings.get_size_of_embeddings())
    store = VectorDB(conf.faiss_config.store.data_path, embedddings.model, index.index)

    rag_model = RAG(store, llm, rag_prompt, conf.faiss_config.search_config.retriever)

    logger.info(rag_model.invoke("Как дела?"))
