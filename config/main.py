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