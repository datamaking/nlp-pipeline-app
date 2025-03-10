# storage/bridge.py
from abc import ABC, abstractmethod


class VectorStorage(ABC):
    @abstractmethod
    def store(self, embeddings: list, metadata: dict):
        pass


class PineconeStorage(VectorStorage):
    def __init__(self, api_key: str, index_name: str):
        import pinecone
        pinecone.init(api_key=api_key)
        self.index = pinecone.Index(index_name)

    def store(self, embeddings: list, metadata: dict):
        vectors = [(str(i), emb.tolist(), metadata)
                   for i, emb in enumerate(embeddings)]
        self.index.upsert(vectors=vectors)


class FAISSStorage(VectorStorage):
    def __init__(self, dimension: int = 768):
        import faiss
        self.index = faiss.IndexFlatL2(dimension)

    def store(self, embeddings: list, metadata: dict):
        self.index.add(np.array(embeddings))