from langchain_openai import OpenAIEmbeddings
from ...base_classes.embedding_base import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    embedding_model: OpenAIEmbeddings

    class Factory:
        def create(self, embedding_model_name: str):
            openai_embedding_model = OpenAIEmbeddingModel(
                embedding_model=OpenAIEmbeddings(model=embedding_model_name)
            )
            return openai_embedding_model

    def generate_embedding(self, text: str) -> list[float]:
        return self.embedding_model.embed_query(text)
