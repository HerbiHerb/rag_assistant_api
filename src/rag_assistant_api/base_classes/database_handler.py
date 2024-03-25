from abc import abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import Any, Iterable, List, Tuple
from ..data_processing.config_schemas import DataProcessingConfig


class DatabaseHandler(BaseModel):
    data_processing_config: DataProcessingConfig

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
