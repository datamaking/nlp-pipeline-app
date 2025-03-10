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