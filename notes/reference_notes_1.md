Here's a comprehensive implementation using design patterns with complete code examples and best practices for your configuration-driven NLP ETL pipeline:

---

### **1. Configuration Structure with Factory Pattern**
```python
# config/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BaseConfig(ABC):
    source_type: str

@dataclass
class FileConfig(BaseConfig):
    path: str
    file_types: list[str]
    recursive: bool = False

@dataclass
class RDBSConfig(BaseConfig):
    connection_string: str
    query: str
    batch_size: int = 1000

# config/__init__.py
class ConfigFactory:
    @staticmethod
    def create(config_dict: dict) -> BaseConfig:
        source_type = config_dict["source_type"]
        if source_type == "file":
            return FileConfig(**config_dict)
        elif source_type == "rdbms":
            return RDBSConfig(**config_dict)
        # Add other sources
        raise ValueError(f"Unknown source type: {source_type}")
```

---

### **2. Pipeline Components with Strategy Pattern**
#### **Data Ingestion**
```python
# ingestion/strategies.py
from abc import ABC, abstractmethod

class IngestionStrategy(ABC):
    @abstractmethod
    def ingest(self, config: BaseConfig):
        pass

class FileIngestion(IngestionStrategy):
    def ingest(self, config: FileConfig):
        from pathlib import Path
        # Implement file loading logic
        return [p.read_text() for p in Path(config.path).rglob('*') 
               if p.suffix.lower() in config.file_types]

class RDBSIngestion(IngestionStrategy):
    def ingest(self, config: RDBSConfig):
        import sqlalchemy
        engine = sqlalchemy.create_engine(config.connection_string)
        return [row for row in engine.execute(config.query)]
```

---

#### **NLP Preprocessing**
```python
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
```

---

### **3. Chunking with Composite Pattern**
```python
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
        return [text[i:i+self.chunk_size] 
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
```

---

### **4. Embeddings with Adapter Pattern**
```python
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
```

---

### **5. Vector Storage with Bridge Pattern**
```python
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
```

---

### **6. Complete Pipeline with Template Method**
```python
# pipeline/core.py
class NLPPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.logger = self._configure_logging()
        
    def run(self):
        try:
            data = self._ingest()
            processed = self._process(data)
            chunks = self._chunk(processed)
            embeddings = self._embed(chunks)
            self._store(embeddings)
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _ingest(self):
        config = ConfigFactory.create(self.config["ingestion"])
        strategy = self._get_ingestion_strategy(config.source_type)
        return strategy.ingest(config)

    def _process(self, data):
        processor = TextProcessorFactory.create(self.config["processing"])
        return [processor.process(text) for text in data]

    def _chunk(self, processed_data):
        chunker = ChunkerFactory.create(self.config["chunking"])
        return [chunker.chunk(text) for text in processed_data]

    def _embed(self, chunks):
        embedder = EmbeddingFactory.create(self.config["embedding"])
        return [embedder.embed(chunk) for chunk in chunks]

    def _store(self, embeddings):
        storage = StorageFactory.create(self.config["storage"])
        storage.store(embeddings)
```

---

### **7. What NOT to Do**
#### **Anti-Patterns to Avoid**
1. **Monolithic Configuration**
   ```python
   # BAD: Hardcoded configuration
   class BadConfig:
       FILE_PATH = "/data"
       DB_CONN = "mysql://..."
       CHUNK_SIZE = 512  # Mixes different concerns
   ```

2. **Tight Coupling**
   ```python
   # BAD: Direct instantiation
   class BadProcessor:
       def __init__(self):
           self.tokenizer = SpacyTokenizer()  # Hardcoded dependency
   ```

3. **Inadequate Error Handling**
   ```python
   # BAD: Silent failures
   def bad_ingest():
       try:
           # ingestion logic
       except:
           pass  # Swallowing exceptions
   ```

4. **Memory Ignorance**
   ```python
   # BAD: Loading everything in memory
   def bad_processing():
       with open("huge_file.txt") as f:
           return f.read()  # Risk of OOM errors
   ```

---

### **8. Best Practices Implementation**
#### **Dependency Injection**
```python
class ProcessingPipeline:
    def __init__(self, 
                 tokenizer: Tokenizer,
                 stemmer: Stemmer,
                 normalizer: Normalizer):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.normalizer = normalizer

    def process(self, text: str):
        return self.normalizer.normalize(
            self.stemmer.stem(
                self.tokenizer.tokenize(text)
            )
        )
```

#### **Optimized Spark Processing**
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

class SparkNLPProcessor:
    def __init__(self, spark_session):
        self.spark = spark_session
        
    def process(self, df: DataFrame, text_column: str):
        tokenize_udf = udf(SpacyTokenizer().tokenize, ArrayType(StringType()))
        return df.withColumn("tokens", tokenize_udf(col(text_column)))
```

---

### **9. Full Configuration Example**
```python
# config/main.py
nlp_config = {
    "ingestion": {
        "source_type": "file",
        "path": "/data/documents",
        "file_types": [".txt", ".docx"],
        "recursive": True
    },
    "processing": {
        "tokenizer": "spacy",
        "stemmer": "porter",
        "lemmatizer": "disabled"
    },
    "chunking": {
        "strategy": "composite",
        "components": [
            {"type": "semantic", "model": "bert-base"},
            {"type": "fixed_size", "size": 256}
        ]
    },
    "embedding": {
        "type": "st5",
        "model": "google/st5-large",
        "batch_size": 32
    },
    "storage": {
        "type": "pinecone",
        "api_key": "env.PINECONE_KEY",
        "index": "doc-vectors"
    }
}
```

---

### **10. Execution Flow**
```python
if __name__ == "__main__":
    pipeline = NLPPipeline(nlp_config)
    
    # Add observers for monitoring
    pipeline.attach(LoggingObserver())
    pipeline.attach(MetricsObserver())
    
    # Execute with retry logic
    retry(pipeline.run, retries=3, delay=10)
```

---

### **Key Libraries to Use**
1. **Data Ingestion**: 
   - `python-docx` for Word documents
   - `BeautifulSoup` for HTML parsing
   - `PySpark` for distributed processing

2. **NLP Processing**:
   - `spaCy` for advanced tokenization
   - `nltk` for stemming/lemmatization
   - `transformers` for sentence embeddings

3. **Vector Stores**:
   - `pinecone-client` for managed vector DB
   - `faiss` for local vector storage
   - `chromadb` for open-source alternative

---

This architecture provides:
- **100% configuration-driven** workflow through Python classes
- **Extensibility** through design patterns
- **Horizontal scalability** via PySpark
- **Production-grade** error handling and monitoring
- **Vendor-agnostic** implementations for all components

The complete implementation would require additional error handling, logging, and validation code, but this structure provides the core framework following software engineering best practices.
