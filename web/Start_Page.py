import streamlit as st

import sys

sys.path.append("./")

from core.vectordb.index import Index
from core.vectordb.vectore_store import VectorDB
from core.llm.rag_pipeline import RAG
from core.llm.model import LocalLLM, OpenAILLM
from core.embeddings.model import LocalEmbeddings
from core.prompts.llm_prompt import rag_prompt
from omegaconf import OmegaConf

st.title("🦜🔗 Начальная страница. Демо")


@st.cache_resource
def load_objects():
    conf = OmegaConf.load("./config/config.yaml")

    # llm = LocalLLM(conf.llm.ollama_config)
    llm = OpenAILLM(conf.llm.chatgpt_config)
    embedddings = LocalEmbeddings(conf.embeddings.ollama_config)
    index = Index(embedddings.get_size_of_embeddings())
    store = VectorDB(conf.faiss_config.store.data_path, embedddings.model, index.index)

    rag_model = RAG(store, llm, rag_prompt, conf.faiss_config.search_config.retriever)

    return rag_model


rag_model = load_objects()

with st.form("chat_form"):
    text = st.text_area(
        "Enter text:",
        "Расскажи основные этапы процедуры регистрации лекарственного препарата по правилам ЕАЭС?",
    )

    if st.form_submit_button("Спросить 🧐"):
        st.info(rag_model.invoke(text))
