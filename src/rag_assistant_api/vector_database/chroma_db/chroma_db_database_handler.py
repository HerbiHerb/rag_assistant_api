from typing import Any, Iterable
from pydantic import Extra
import chromadb
import uuid
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from pydantic.main import BaseModel
from ...data_structures.data_structures import ChromaDBConfig, DataProcessingConfig
from ...base_classes.database_handler import DatabaseHandler


class ChromaDatabaseHandler(DatabaseHandler, extra=Extra.allow):
    db_config: ChromaDBConfig

    class Factory:
        def create(self, db_config_Data: dict, data_processing_config: BaseModel):
            chroma_db_config = ChromaDBConfig(**db_config_Data["chroma_db"])
            database_handler = ChromaDatabaseHandler(
                db_config=chroma_db_config,
                data_processing_config=data_processing_config,
            )
            return database_handler

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.db_config.chroma_db_fp,
            )
        )
        try:
            self.collection = self.client.get_collection(
                name=self.db_config.collection_name,
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    model_name=self.data_processing_config.embedding_model
                ),
            )
        except Exception as e:
            print(e)

    def create_database(self) -> None:
        try:
            # Initialize chromadb client
            self.client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.db_config.chroma_db_fp,
                )
            )
            self.collection = self.client.create_collection(
                self.db_config.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    model_name=self.data_processing_config.embedding_model
                ),
            )
            test = 0
        except Exception as e:
            print(e)

    def delete_database(self) -> None:
        """Deletes the database"""
        # TODO: Implementation of database removing
        print("Database deletion needs to be implemented")

    def query(
        self, embedding: Iterable, top_k: int, filter: dict = None
    ) -> tuple[list[str]]:
        doc_search_res = self.collection.query(
            query_embeddings=embedding, n_results=top_k
        )
        result_texts = [
            search_res["text"] for search_res in doc_search_res["metadatas"][0]
        ]
        return result_texts, doc_search_res["metadatas"][0]

    def upsert(self, data: list[dict[str, Any]]) -> None:
        embeddings = [doc_data["values"] for doc_data in data]
        meta_data = [doc_data["metadata"] for doc_data in data]
        ids = [str(uuid.uuid4()) for idx in range(len(embeddings))]
        self.collection.add(embeddings=embeddings, metadatas=meta_data, ids=ids)
