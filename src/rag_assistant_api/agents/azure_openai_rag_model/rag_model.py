import os
import yaml
from typing import List, Dict, Optional, Callable
import openai
import pinecone
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from ...base_classes.agent_base import AgentBase
from ..langchain_tools.tools import DocumentSearch
from ...data_processing.config_schemas import PineconeConfig, DataProcessingConfig
from ...data_processing.pinecone.pinecone_database_handler import (
    PineconeDatabaseHandler,
)


def initialize_rag_model(model_name: str, embedding_model: str):
    with open(os.environ["VECTOR_DB_CONFIG_FP"], "r") as file:
        config_data = yaml.safe_load(file)
    data_processing_config = DataProcessingConfig(**config_data["data_processing"])
    pinecone_config = PineconeConfig(**config_data["pinecone_db"])
    database_handler = PineconeDatabaseHandler(
        index=pinecone.Index(pinecone_config.index_name),
        data_processing_config=data_processing_config,
        pinecone_config=PineconeConfig(**config_data["pinecone_db"]),
    )
    embedding_model = OpenAIEmbeddings(model=embedding_model)
    document_search_tool = DocumentSearch(
        embedding_model=embedding_model, database_handler=database_handler
    )
    rag_agent = OpenAIRAGModel(model_name=model_name, search_tool=document_search_tool)
    return rag_agent


class AzureOpenAIRAGModel(AgentBase):
    def __init__(self, model_name: str, search_tool: Callable) -> None:
        self.model_name = model_name
        self.search_tool = search_tool

    def run(self, query: str, chat_messages: List[Dict[str, str]]):
        result_texts, result_metas = self.search_tool(query)
        combined_result_text = " ".join(result_texts)
        tmp_user_msg = (
            query
            + f"\n\nZur Beantwortung der oben stehenden Kundenfrage, stehen dir die folgenden Kontextinformationen zur Verfügung und halte dich strikt an die Kontextinformationen:\n{combined_result_text}"
        )
        chat_messages.append({"role": "user", "content": tmp_user_msg})
        completion = openai.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            temperature=0.0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        return completion.choices[0].message.content, result_texts, result_metas


class OpenAIRAGModel(AgentBase):
    def __init__(self, model_name: str, search_tool: Callable) -> None:
        self.model_name = model_name
        self.search_tool = search_tool

    def run(self, query: str, chat_messages: List[Dict[str, str]]):
        result_texts, result_metas = self.search_tool(query)
        combined_result_text = " ".join(result_texts)
        tmp_user_msg = (
            query
            + f"\n\nZur Beantwortung der oben stehenden Kundenfrage, stehen dir die folgenden Kontextinformationen zur Verfügung und halte dich strikt an die Kontextinformationen:\n{combined_result_text}"
        )
        chat_messages.append({"role": "user", "content": tmp_user_msg})
        completion = openai.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            temperature=0.0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        return completion.choices[0].message.content, result_texts, result_metas
