import pathlib
from typing import Literal, Optional

from pydantic import BaseModel, Extra, Field
from pydantic.types import StrictInt


class DataProcessingConfig(BaseModel, extra=Extra.forbid, allow_mutation=False):
    data_folder_fp: str = Field(default=None, description="Path to where txt files are stored.")
    batch_size: StrictInt = Field(default=100, description="Batch size of vectors to be upsert together.")
    chunk_size: StrictInt = Field(default=256, description="Chunk size of tokens.")
    overlap: StrictInt = Field(default=20, description="Chunk overlap.")
    embedding_model: Literal["text-embedding-ada-002"] = Field(
        default="text-embedding-ada-002",
        description="Embedding model from AzureOpenAI. It is storongly advised to use ada-002. "
        + "Make sure you use the Model Implementation Name from AzureOpenAI.",
    )
    embeddings_file_path: Optional[pathlib.Path] = Field(
        default=None, description="Path to where embeddings are saved in a csv file."
    )
    config_credentials: Optional[pathlib.Path] = Field(default=None, description="Path to where configs are stored.")
    meta_prefix: Optional[str] = Field(default='', description="Prefix under witch Meta Files are found.")


class PineconeConfig(BaseModel, extra=Extra.forbid, allow_mutation=False):
    index_name: str = Field(description="Name of index used to save the embeddings.")
    dimension: StrictInt = Field(default=1536, description="Vector deimension.")
    metric: Literal["cosine", "dotproduct", "euclidean"] = Field(
        default="cosine", description="Metric to be used to find similar vectors."
    )
    top_k: StrictInt = Field(default=5, description="Number of results for the semantic search.")
    config_credentials: Optional[pathlib.Path] = Field(default=None, description="Path to where configs are stored.")


class ChromadbConfig(BaseModel, extra=Extra.forbid, allow_mutation=False):
    collection_name: str = Field(
        default="llm_demo_documents",
        description="ChromaDB collection name in which the vectors are stored",
    )
    data_folder_fp: str = Field(default=None, description="Path to where txt files are stored.")
    config_credentials: Optional[pathlib.Path] = Field(default=None, description="Path to where configs are stored.")
    persist_directory: str = Field(default=None, description="Path to where the chromadb is persisted.")
    embedding_model: Literal["text-embedding-ada-002"] = Field(
        default="text-embedding-ada-002",
        description="Embedding model from AzureOpenAI. It is storongly advised to use ada-002. "
        + "Make sure you use the Model Implementation Name from AzureOpenAI.",
    )
    max_token_size: StrictInt = Field(default=256, description="Chunk size of tokens.")
    tokenizer_model: Literal["text-davinci-003"] = Field(
        default="text-davinci-003", description="Tokenizer model for text tokenization "
    )
