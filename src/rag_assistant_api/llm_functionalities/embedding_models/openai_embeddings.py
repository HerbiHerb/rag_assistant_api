from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from ...base_classes.embedding_base import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    embedding_model: OpenAIEmbeddings

    class Factory:
        def create(self, llm_service: str, embedding_model_name: str):
            if llm_service == "openai":
                return OpenAIEmbeddingModel(
                    embedding_model=OpenAIEmbeddings(model=embedding_model_name)
                )
            elif llm_service == "azure":
                return OpenAIEmbeddingModel(
                    embedding_model=AzureOpenAIEmbeddings(model=embedding_model_name)
                )
            else:
                raise ValueError(
                    "If you use OpenAIEmbeddingModel the llm_service variable should be either openai or azure!!"
                )

    def generate_embedding(self, text: str) -> list[float]:
        return self.embedding_model.embed_query(text)
