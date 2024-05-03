from langchain.tools import BaseTool
from typing import Tuple, List
from langchain_openai import OpenAIEmbeddings
from ...utils.data_processing_utils import get_embedding
from ...base_classes.database_handler import DatabaseHandler


class DocumentSearch(BaseTool):
    name = "document_search"
    description = "Useful for when you need to answer topic-specific questions about a law, finance or insurance"
    embedding_model: OpenAIEmbeddings
    database_handler: DatabaseHandler

    def _run(self, query: str) -> Tuple[List[str]]:
        """Use the tool"""
        query_embeddings = get_embedding(query, embedding_model=self.embedding_model)
        result_texts, result_files = self.database_handler.query(
            embedding=query_embeddings,
            top_k=self.database_handler.pinecone_config.top_k,
        )
        return result_texts, result_files
