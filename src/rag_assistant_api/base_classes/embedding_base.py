from pydantic.v1 import BaseModel
from abc import abstractmethod


class EmbeddingModel(BaseModel):

    @abstractmethod
    def generate_embedding(text: str) -> list[float]:
        pass
