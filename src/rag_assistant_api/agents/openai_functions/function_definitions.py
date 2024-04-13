from pydantic import BaseModel, Extra, Field
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from ...utils.data_processing_utils import get_embedding
from ...vector_database.pinecone.pinecone_database_handler import (
    PineconeDatabaseHandler,
)


class PineconeDocumentSearch(BaseModel, extra=Extra.allow):
    name = "document_search"
    description = "Useful if you need facts from documents to answer a user question. All documents are considered to search for information."
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
        self.meta_data.extend(result_files)
        return result_text


DOCUMENT_SEARCH = {
    "type": "function",
    "function": {
        "name": "document_search",
        "description": """This function searches for relevant information from a vector database.
        The purpose of this function is to provide grounded information to an AI system to answer the user questions. """,
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


class PineconeDocumentFilterSearch(BaseModel, extra=Extra.allow):
    name = "document_filter_search"
    description = "Useful if you need facts from a specific document to answer a user question. Information only from this document are considered."
    embedding_model: OpenAIEmbeddings
    database_handler: PineconeDatabaseHandler
    filter: dict = Field(
        default=None, description="Filter dictionary for the vector search"
    )
    meta_data: list[dict[str, str]] = Field(
        default=[], description="Field to save meta data of the document search"
    )

    def __call__(self, search_string: str, document_name: str = None) -> str:
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
        self.meta_data.extend(result_files)
        return result_text


DOCUMENT_FILTER_SEARCH = {
    "type": "function",
    "function": {
        "name": "document_filter_search",
        "description": """This function searches for relevant chunks of a document from a vector database. 
        Use this function if the user mentioned the document name and you think you need only small relevant text snippets from 
        the document to answer the user query. Here is one possible example query for this function: 
        Kannst du mir sagen was im Buch <Buchname> über das Thema <Spezifisches_Thema> steht?""",
        "parameters": {
            "type": "object",
            "properties": {
                "search_string": {
                    "type": "string",
                    "description": "Search string to execute a semantic search on a vector database.",
                },
                "document_name": {
                    "type": "string",
                    "description": "(Optional) Useful if the user mentioned the name of the document in the query. If the name is not mentoined, ignore this parameter.",
                },
            },
            "required": ["search_string", "document_name"],
        },
    },
}


class DocumentAnalyzer(BaseModel, extra=Extra.allow):
    name = "document_analyzer"
    description = "Useful if you need to find the right document based on the user query and extract information from the whole document. Not only snippets from it. Use this tool if you think you need to consider the whole document to answer th user query or if the user asks particularly for it."
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
        self.meta_data.extend(result_files)
        return result_text


DOCUMENT_ANALYZER = {
    "type": "function",
    "function": {
        "name": "document_analyzer",
        "description": """Useful if you need to find the right document based on the user query and extract information from the whole document. Not only snippets from it. 
        Use this tool if you think you need to consider the whole document to answer th user query or if the user asks particularly for it.""",
        "parameters": {
            "type": "object",
            "properties": {
                "search_string": {
                    "type": "string",
                    "description": "Search string to execute a semantic search on a vector database to extract information from a document.",
                },
            },
            "required": ["search_string"],
        },
    },
}


class VectorFilterSearchTool(
    BaseModel, extra=Extra.allow, arbitrary_types_allowed=True
):
    name = "vector_filter_search"
    description = "Useful if you need facts to answer a user question."
    embedding_model: OpenAIEmbeddings
    openai_client: OpenAI

    def __call__(self, search_str: str, filter_field: str, filter_value: str) -> str:
        result_text = ""
        return result_text


class SummarizationTool(BaseModel, extra=Extra.allow, arbitrary_types_allowed=True):
    name = "text_summarization"
    description = "Useful if you need facts to answer a user question."
    embedding_model: OpenAIEmbeddings
    openai_client: OpenAI

    def __call__(self, chapter: str, document_id: str) -> str:
        result_text = ""
        return result_text


class GetUserInformation(BaseModel, extra=Extra.allow, arbitrary_types_allowed=True):
    name = "get_user_information"
    description = "Useful if you need information about the user."
    embedding_model: OpenAIEmbeddings
    openai_client: OpenAI

    def __call__(self, user_id: str) -> str:
        result_text = ""
        return result_text
