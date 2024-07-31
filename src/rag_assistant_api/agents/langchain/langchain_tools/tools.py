from langchain.tools import BaseTool
from typing import Tuple, List
from ....base_classes.database_handler import DatabaseHandler
from ....base_classes.embedding_base import EmbeddingModel
from ....utils.data_processing_utils import get_embedding


class DocumentSearch(BaseTool):
    name = "document_search"
    description = """Useful if you need to search for relevant information to answer the user query.

    Args:
        query: The search query for the vector database. 
    """
    embedding_model: EmbeddingModel
    database_handler: DatabaseHandler

    def _run(self, query: str) -> Tuple[List[str]]:
        """Use the tool"""
        query_embeddings = get_embedding(query, embedding_model=self.embedding_model)
        vecdb_retr_data = self.database_handler.query(
            embedding=query_embeddings,
            top_k=self.database_handler.db_config.top_k,
        )
        return vecdb_retr_data.chunk_texts, vecdb_retr_data.meta_data


class SQLQuerySearch(BaseTool):
    name = "sql_query"
    description = """"Useful if you need to get data from an sql database. The data table is called 'cp_dwh'.

    Args:
        sql_query: The search query for the vector database. 
    """
    embedding_model: EmbeddingModel
    database_handler: DatabaseHandler

    def _run(self, sql_query: str) -> Tuple[List[str]]:
        """Use the tool"""
        test = 0
        return "SQL-Answer"
