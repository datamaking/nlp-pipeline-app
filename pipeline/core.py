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