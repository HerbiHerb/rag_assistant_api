import os
from typing import Any, Iterable
from pydantic import BaseModel
from ...data_structures.data_structures import PineconeConfig, DataProcessingConfig
import pinecone
from pinecone import Pinecone
from ...base_classes.database_handler import DatabaseHandler


class PineconeDatabaseHandler(DatabaseHandler):
    db_config: PineconeConfig

    class Factory:
        def create(self, db_config_Data: dict, data_processing_config: BaseModel):
            pinecone_config = PineconeConfig(
                api_key=os.getenv("PINECONE_API_KEY"), **db_config_Data["pinecone_db"]
            )
            database_handler = PineconeDatabaseHandler(
                data_processing_config=data_processing_config,
                db_config=pinecone_config,
            )
            return database_handler

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pinecone_instance = Pinecone(api_key=self.db_config.api_key)
        self.index = self.pinecone_instance.Index(self.db_config.index_name)

    def create_database(self) -> None:
        try:
            pinecone.create_index(
                name=self.db_config.index_name,
                dimension=self.db_config.dimension,
                metric=self.db_config.metric,
            )
            self.index = pinecone.Index(index_name=self.db_config.index_name)
        except Exception as e:
            print(e)

    def delete_database(self) -> None:
        try:
            pinecone.delete_index(self.db_config.index_name)
        except Exception as e:
            print("No index available for deletion")

    def query(
        self, embedding: Iterable, top_k: int, filter: dict = None
    ) -> tuple[list[str]]:
        query_results = self.index.query(
            vector=embedding,
            filter=filter,
            top_k=top_k,
            include_metadata=True,
        )
        result_texts = [
            search_res["metadata"]["text"] for search_res in query_results["matches"]
        ]
        result_meta = [
            search_res["metadata"] for search_res in query_results["matches"]
        ]
        return result_texts, result_meta

    def upsert(self, data: list[dict[str, Any]]) -> None:
        self.index.upsert(vectors=data)
