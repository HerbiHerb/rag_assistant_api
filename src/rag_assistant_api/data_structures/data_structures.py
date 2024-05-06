import pathlib
import os
from typing import Literal, Optional, Any
from pydantic import BaseModel, Extra, Field, validator, root_validator
from pydantic.networks import import_email_validator
from pydantic.types import StrictInt
from ..utils.file_loading import load_yaml_file


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
    api_key: str = Field(description="Field whitch saves the api key.")
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


class ChromaDBConfig(BaseModel, extra=Extra.forbid, allow_mutation=False):
    collection_name: str = Field(description="The name of the database")
    chroma_db_fp: str = Field(description="Path to the chroma database")
    dimension: StrictInt = Field(default=1536, description="Vector deimension.")
    metric: Literal["cosine", "dotproduct", "euclidean"] = Field(
        default="cosine", description="Metric to be used to find similar vectors."
    )
    top_k: StrictInt = Field(
        default=5, description="Number of results for the semantic search."
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


class ConfigFileValidator(BaseModel):
    usage_settings: dict
    data_processing_config: DataProcessingConfig
    document_processing_config: DocumentProcessingConfig
    chroma_db_config: ChromaDBConfig
    pinecone_db_config: PineconeConfig
    prompt_configs_fp: str

    @staticmethod
    def _check_agent_prompt_config(agent_prompt_config: str, prompt_configs_fp: str):
        try:
            prompt_configs = load_yaml_file(prompt_configs_fp)
        except Exception as e:
            print(e)
            raise ValueError(f"{prompt_configs_fp} is not a valid path to a yaml file")
        assert (
            agent_prompt_config in prompt_configs
        ), f"{agent_prompt_config} is not in the prompt configs file"
        for prompt_config in prompt_configs:
            assert (
                "system_msg" in prompt_configs[prompt_config]
            ), f"system_msg not in {prompt_config}"

    @root_validator
    def check_variables(cls, values):
        usage_settings = values.get("usage_settings")
        assert "agent_type" in usage_settings, "agent_type is not in usage_settings"
        assert (
            "agent_prompt_config" in usage_settings
        ), "agent_prompt_config is not in usage_settings"
        assert "vector_db" in usage_settings, "vector_db is not in usage_settings"
        assert "llm_service" in usage_settings, "llm_service is not in usage_settings"
        assert (
            "function_call_prefix" in usage_settings
        ), "function_call_prefix is not in usage_settings"

        ConfigFileValidator._check_agent_prompt_config(
            agent_prompt_config=usage_settings["agent_prompt_config"],
            prompt_configs_fp=os.getenv("PROMPT_CONFIGS_FP"),
        )
        return values


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
    final_answer: str = Field(
        default="", description="The final answer of the assistant."
    )
    function_responses: list[Any] = Field(
        default=[],
        description="All information of the function responses to the final answer.",
    )
    chat_messages: list[dict[str, str]] = Field(
        default=[], description="Current chat messages of the conversation."
    )

    def add_function_response(self, new_response: str) -> None:
        self.function_responses.append(new_response)
