import os
import openai
from pydantic.types import confloat
import yaml
from dotenv import load_dotenv
import pytest
from rag_assistant_api.data_structures.data_structures import VectorDBRetrievalData
from rag_assistant_api.vector_database.vector_db_factory import VectorDBFactory
from rag_assistant_api.llm_functionalities.embedding_models.embedding_model_factory import (
    EmbeddingModelFactory,
)
from rag_assistant_api.utils.data_processing_utils import get_embedding


def test_vector_db_retrieval():
    load_dotenv()
    with open(
        os.environ["CONFIG_FP"],
        "r",
    ) as file:
        config_data = yaml.safe_load(file)
    if config_data["usage_settings"]["llm_service"] == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
    database_handler = VectorDBFactory.create_vector_db_instance(
        vector_db_cls=config_data["usage_settings"]["vector_db"],
        config_data=config_data,
    )
    embedding_model = EmbeddingModelFactory.create_embedding_model(
        embedding_model_cls=config_data["usage_settings"]["embeddding_model_cls"],
        embedding_model_name=config_data["language_models"]["embedding_model"],
    )
    query_embedding = get_embedding(
        text="Das ist ein Test", embedding_model=embedding_model
    )
    vecdb_retr_data = database_handler.query(
        embedding=query_embedding,
        filter=None,
        top_k=database_handler.db_config.top_k,
    )
    assert isinstance(vecdb_retr_data, VectorDBRetrievalData)
