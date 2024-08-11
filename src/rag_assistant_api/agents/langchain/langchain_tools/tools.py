from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Tuple, List, Type
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
        return vecdb_retr_data.meta_data


class DocumentFilterSearchInput(BaseModel):
    search_string: str = Field(
        description="The search string to search for relevant information in the vector database. "
    )
    document_name: str = Field(
        description="The name of the document if the user has mentioned it in the search query."
    )


class DocumentFilterSearch(BaseTool):
    name = "document_filter_search"
    description = """USe this if you need to search for relevant information inside a specific document to answer the user query.
    """
    embedding_model: EmbeddingModel
    database_handler: DatabaseHandler
    args_schema: Type[BaseModel] = DocumentFilterSearchInput

    def _run(self, search_string: str, document_name: str) -> Tuple[List[str]]:
        """Use the tool"""
        query_embeddings = get_embedding(
            search_string, embedding_model=self.embedding_model
        )
        vecdb_retr_data = self.database_handler.query(
            embedding=query_embeddings,
            top_k=self.database_handler.db_config.top_k,
        )
        return vecdb_retr_data.meta_data


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
