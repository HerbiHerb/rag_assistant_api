from abc import abstractmethod
from pydantic import BaseModel, Extra
from typing import Any, Iterable, List, Tuple
from ..data_structures.data_structures import DataProcessingConfig
from ..data_structures.data_structures import (
    DataProcessingConfig,
)
from ..data_structures.data_structures import VectorDBRetrievalData


class DatabaseHandler(BaseModel, arbitrary_types_allowed=True, extra=Extra.allow):
    data_processing_config: DataProcessingConfig

    @abstractmethod
    def create_database() -> None:
        pass

    @abstractmethod
    def delete_database() -> None:
        pass

    @abstractmethod
    def query(embedding: Iterable, top_k: int) -> VectorDBRetrievalData:
        pass

    @abstractmethod
    def get_all_document_meta_data(self) -> list[dict[str, str]]:
        pass

    @abstractmethod
    def upsert(data: Any) -> None:
        pass
