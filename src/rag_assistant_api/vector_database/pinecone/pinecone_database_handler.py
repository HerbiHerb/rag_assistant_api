import os
from typing import Any, Iterable
from pydantic.main import BaseModel
from ...data_structures.data_structures import PineconeConfig, DataProcessingConfig
import pinecone
from pinecone import Pinecone
from ...base_classes.database_handler import DatabaseHandler
from ...data_structures.data_structures import VectorDBRetrievalData


class PineconeDatabaseHandler(DatabaseHandler):
    # index: pinecone.Index
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
    ) -> VectorDBRetrievalData:
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
        vecdb_retr_data = VectorDBRetrievalData(
            chunk_texts=result_texts, meta_data=result_meta
        )
        return vecdb_retr_data

    def _get_all_records(self, ids) -> list[Any]:
        records = []
        for i in range(0, len(ids), 1000):
            res = self.index.fetch(ids[i : i + 1000])
            for record in res["vectors"].values():
                records.append(record)
        return records

    def _extract_metadata(self, records) -> list[dict[str, str]]:
        metadata_list = []
        for rec in records:
            try:
                metadata = rec["metadata"]
                metadata_list.append(metadata)
            except:
                pass
        return metadata_list

    def get_all_document_meta_data(self) -> list[dict[str, str]]:
        all_metadata = []
        pagination_token = None
        while True:
            results = self.index.list_paginated(
                limit=99, pagination_token=pagination_token
            )

            ids = [v.id for v in results.vectors]
            records = self._get_all_records(ids)
            metadata = self._extract_metadata(records)
            all_metadata.extend(metadata)
            if not results.pagination:
                break
            pagination_token = results.pagination.next
        cleaned_metadata = [
            dict(t)
            for t in {
                tuple(
                    [
                        ("document_name", entry["document_name"]),
                        ("document_type", entry["type"]),
                        ("document_genre", entry["genre"]),
                    ]
                )
                for entry in all_metadata
            }
        ]

        return cleaned_metadata

    def upsert(self, data: list[dict[str, Any]]) -> None:
        self.index.upsert(vectors=data)
