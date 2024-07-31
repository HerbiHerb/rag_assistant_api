from langchain_openai import OpenAIEmbeddings
from ...llm_functionalities.embedding_models.openai_embeddings import (
    OpenAIEmbeddingModel,
)


class EmbeddingModelFactory:
    factories = {}

    @staticmethod
    def create_embedding_model(
        embedding_model_cls: str,
        llm_service: str,
        embedding_model_name: str,
    ):
        if not embedding_model_cls in EmbeddingModelFactory.factories:
            try:
                EmbeddingModelFactory.factories[embedding_model_cls] = eval(
                    embedding_model_cls + ".Factory()"
                )
            except NameError as e:
                raise NameError(
                    "NameError: Please define one of the following embedding model types in the config.yaml file for embeddding_model_cls: OpenAIEmbeddingModel"
                )

        return EmbeddingModelFactory.factories[embedding_model_cls].create(
            llm_service, embedding_model_name
        )
