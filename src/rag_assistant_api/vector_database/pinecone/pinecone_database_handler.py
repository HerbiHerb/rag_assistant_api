from typing import Any, Iterable
from ..config_schemas import PineconeConfig, DataProcessingConfig
import pinecone
from ...base_classes.database_handler import DatabaseHandler
import pinecone


class PineconeDatabaseHandler(DatabaseHandler):
    index: pinecone.Index
    pinecone_config: PineconeConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_database(self) -> None:
        try:
            pinecone.create_index(
                name=self.pinecone_config.index_name,
                dimension=self.pinecone_config.dimension,
                metric=self.pinecone_config.metric,
            )
            self.index = pinecone.Index(index_name=self.pinecone_config.index_name)
        except Exception as e:
            print(e)

    def delete_database(self) -> None:
        try:
            pinecone.delete_index(self.pinecone_config.index_name)
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
