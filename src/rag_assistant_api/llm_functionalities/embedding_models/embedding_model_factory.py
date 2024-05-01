from langchain_openai import OpenAIEmbeddings
from ...llm_functionalities.embedding_models.openai_embeddings import (
    OpenAIEmbeddingModel,
)


def create_embedding_model(llm_service: str, model: str):
    if llm_service == "openai":
        return OpenAIEmbeddingModel(embedding_model=OpenAIEmbeddings(model=model))
    else:
        raise ValueError(llm_service)
