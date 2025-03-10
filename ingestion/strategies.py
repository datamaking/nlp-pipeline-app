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