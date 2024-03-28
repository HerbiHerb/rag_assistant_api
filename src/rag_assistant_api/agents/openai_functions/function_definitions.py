import os
from openai import AzureOpenAI
from pydantic import BaseModel, Extra, Field
from typing import Tuple, List
from langchain_openai import OpenAIEmbeddings
from ...data_processing.data_processing_utils import get_embedding
from ...base_classes.database_handler import DatabaseHandler
from ...data_processing.pinecone.pinecone_database_handler import (
    PineconeDatabaseHandler,
)


class PineconeDocumentSearch(BaseModel, extra=Extra.allow):
    name = "document_search"
    description = "Useful for when you need facts to answer a user question."
    embedding_model: OpenAIEmbeddings
    database_handler: PineconeDatabaseHandler
    filter: dict = Field(
        default=None, description="Filter dictionary for the vector search"
    )
    meta_data: list[dict[str, str]] = Field(
        default=[], description="Field to save meta data of the document search"
    )

    def __call__(self, search_string: str) -> str:
        """Use the tool"""
        query_embeddings = get_embedding(
            search_string, embedding_model=self.embedding_model
        )
        result_texts, result_files = self.database_handler.query(
            embedding=query_embeddings,
            filter=self.filter,
            top_k=self.database_handler.pinecone_config.top_k,
        )
        result_text = "\n\n".join(result_texts)
        self.meta_data = result_files
        return result_text


DOCUMENT_SEARCH = {
    "type": "function",
    "function": {
        "name": "fetch_relevant_information",
        "description": "This function searches for relevant information from a vector database. The purpose of this function is to provide grounded information to an AI system to answer the user questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_string": {
                    "type": "string",
                    "description": "Search string to execute a semantic search on a vector database.",
                },
            },
            "required": ["search_string"],
        },
    },
}

TOOLS_LIST = [DOCUMENT_SEARCH]