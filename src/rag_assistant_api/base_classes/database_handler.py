from abc import abstractmethod
from pydantic import BaseModel, ConfigDict, Extra
from typing import Any, Iterable, List, Tuple, Type
import pinecone
from ..data_structures.data_structures import DataProcessingConfig
from ..data_structures.data_structures import (
    PineconeConfig,
    DataProcessingConfig,
    ChromaDBConfig,
)


class DatabaseHandler(BaseModel, arbitrary_types_allowed=True, extra=Extra.allow):
    data_processing_config: DataProcessingConfig

    @abstractmethod
    def create_database() -> None:
        pass

    @abstractmethod
    def delete_database() -> None:
        pass

    @abstractmethod
    def query(embedding: Iterable, top_k: int) -> Tuple[List[str]]:
        pass

    @abstractmethod
    def upsert(data: Any) -> None:
        pass
