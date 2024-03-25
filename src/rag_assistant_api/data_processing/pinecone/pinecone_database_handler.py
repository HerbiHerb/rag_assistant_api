from typing import Any, Iterable, List, Dict, Tuple
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
        pinecone.create_index(name=self.pinecone_config.index_name, dimension=self.pinecone_config.dimension, metric=self.pinecone_config.metric)
        self.index = pinecone.Index(index_name=self.pinecone_config.index_name)

    def delete_database(self) -> None:
        try:
            pinecone.delete_index(self.pinecone_config.index_name)
        except Exception as e:
            print('No index available for deletion')

    def query(self, embedding: Iterable, top_k: int) -> Tuple[List[str]]:
        query_results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
        )
        result_texts = [search_res["metadata"]["text"] for search_res in query_results["matches"]]
        result_files = [search_res["metadata"] for search_res in query_results["matches"]]
        return result_texts, result_files

    def upsert(self, data: List[Dict[str, Any]]) -> None:
        self.index.upsert(vectors=data)
