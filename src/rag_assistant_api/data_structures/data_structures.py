import pathlib
from typing import Literal, Optional

from pydantic import BaseModel, Extra, Field
from pydantic.types import StrictInt


class DataProcessingConfig(BaseModel, extra=Extra.forbid, allow_mutation=False):
    data_folder_fp: str = Field(
        default=None, description="Path to where txt files are stored."
    )
    batch_size: StrictInt = Field(
        default=100, description="Batch size of vectors to be upsert together."
    )
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
    config_credentials: Optional[pathlib.Path] = Field(
        default=None, description="Path to where configs are stored."
    )
    meta_prefix: Optional[str] = Field(
        default="", description="Prefix under witch Meta Files are found."
    )


class PineconeConfig(BaseModel, extra=Extra.forbid, allow_mutation=False):
    index_name: str = Field(description="Name of index used to save the embeddings.")
    dimension: StrictInt = Field(default=1536, description="Vector deimension.")
    metric: Literal["cosine", "dotproduct", "euclidean"] = Field(
        default="cosine", description="Metric to be used to find similar vectors."
    )
    top_k: StrictInt = Field(
        default=5, description="Number of results for the semantic search."
    )
    config_credentials: Optional[pathlib.Path] = Field(
        default=None, description="Path to where configs are stored."
    )


class DocumentProcessingConfig(BaseModel, extra=Extra.forbid):
    meta_data_pattern: str = Field(
        description="Regex pattern to extract meta data from a document"
    )
    part_seperator: str = Field(
        description="String to seperate diferent parts of a document"
    )
    chapter_seperator: str = Field(
        description="String to seperate diferent chapters of a document"
    )
    subchapter_seperator: str = Field(
        description="String to seperate diferent subchapters of a document"
    )
    meta_data_fields: list[str] = Field(
        description="List which defines the different fields of the meta data in a document"
    )


class AgentData(BaseModel):
    openai_key: str
    max_token_number: int
    embedding_model_name: str
    embedding_token_counter: str
    pinecone_key: str
    pinecone_environment: str
    pinecone_index_name: str
    chatmessages_csv_path: str
    listening_sound_path: str


class AgentAnswerData(BaseModel):
    query_msg_idx: int
    final_answer: str = Field(default="")
    function_responses: list[str] = Field(default=[])

    def add_function_response(self, new_response: str) -> None:
        self.function_responses.append(new_response)
