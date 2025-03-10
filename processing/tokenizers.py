# processing/tokenizers.py
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list:
        pass


class SpacyTokenizer(Tokenizer):
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, text: str) -> list:
        return [token.text for token in self.nlp(text)]


class SentenceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)