import faiss


class Index:
    def __init__(self, embedding_size: int) -> None:
        self.index = faiss.IndexFlatL2(embedding_size)