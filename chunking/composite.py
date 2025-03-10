# chunking/composite.py
from abc import ABC, abstractmethod


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        pass


class FixedSizeChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return [text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)]


class CompositeChunker(ChunkingStrategy):
    def __init__(self, strategies: list[ChunkingStrategy]):
        self.strategies = strategies

    def chunk(self, text: str) -> list[str]:
        results = []
        current_text = text
        for strategy in self.strategies:
            chunks = strategy.chunk(current_text)
            results.extend(chunks)
            current_text = " ".join(chunks)
        return results