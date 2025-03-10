# embeddings/adapters.py
from abc import ABC, abstractmethod
import numpy as np


class EmbeddingAdapter(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass


class ST5Adapter(EmbeddingAdapter):
    def __init__(self, model_name: str = "google/st5-base"):
        from transformers import T5Tokenizer, T5Model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5Model.from_pretrained(model_name)

    def embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()


class TFIDFAdapter(EmbeddingAdapter):
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)

    def embed(self, text: str) -> np.ndarray:
        return self.vectorizer.transform([text]).toarray()