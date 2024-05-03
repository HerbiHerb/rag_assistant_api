from langchain_openai import OpenAIEmbeddings
from ...base_classes.embedding_base import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    embedding_model: OpenAIEmbeddings

    def generate_embedding(self, text: str) -> list[float]:
        return self.embedding_model.embed_query(text)
