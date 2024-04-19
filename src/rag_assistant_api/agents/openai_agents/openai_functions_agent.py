from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
import os
import pinecone
import yaml
from copy import deepcopy
import tiktoken
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from ..openai_functions.function_definitions import (
    PineconeDocumentSearch,
    PineconeDocumentFilterSearch,
    DocumentAnalyzer,
    DOCUMENT_SEARCH,
    DOCUMENT_FILTER_SEARCH,
    DOCUMENT_ANALYZER,
)
from ...base_classes.agent_base import OpenAIAgent
from ...data_structures.data_structures import PineconeConfig, DataProcessingConfig
from ...vector_database.pinecone.pinecone_database_handler import (
    PineconeDatabaseHandler,
)
from ...data_structures.data_structures import AgentAnswerData
from ..exceptions import TokenLengthExceedsMaxTokenNumber
from ...utils.agent_utils import count_tokens_of_conversation, get_max_token_number


class OpenAIFunctionsAgent(OpenAIAgent):
    available_functions: dict[str, BaseModel]
    function_definitions: list[dict]

    @classmethod
    def initialize_agent(cls, document_filter: dict = None):
        with open(os.getenv("CONFIG_FP"), "r") as file:
            config_data = yaml.safe_load(file)
        with open(os.getenv("PROMPT_CONFIGS_FP"), "r") as file:
            prompt_configs = yaml.safe_load(file)
        data_processing_config = DataProcessingConfig(**config_data["data_processing"])
        pinecone_config = PineconeConfig(**config_data["pinecone_db"])
        database_handler = PineconeDatabaseHandler(
            index=pinecone.Index(pinecone_config.index_name),
            data_processing_config=data_processing_config,
            pinecone_config=PineconeConfig(**config_data["pinecone_db"]),
        )
        available_functions = {
            "document_search": PineconeDocumentSearch(
                embedding_model=OpenAIEmbeddings(
                    model=config_data["language_models"]["embedding_model"]
                ),
                database_handler=database_handler,
            ),
            "document_filter_search": PineconeDocumentFilterSearch(
                embedding_model=OpenAIEmbeddings(
                    model=config_data["language_models"]["embedding_model"]
                ),
                database_handler=database_handler,
                filter_dict=document_filter,
            ),
            "document_analyzer": DocumentAnalyzer(
                embedding_model=OpenAIEmbeddings(
                    model=config_data["language_models"]["embedding_model"]
                ),
                database_handler=database_handler,
            ),
        }
        function_definitions = [
            DOCUMENT_SEARCH,
            DOCUMENT_FILTER_SEARCH,
            DOCUMENT_ANALYZER,
        ]
        return OpenAIFunctionsAgent(
            openai_client=OpenAI(),
            model_name=config_data["language_models"]["model_name"],
            available_functions=available_functions,
            function_definitions=function_definitions,
            initial_system_msg=prompt_configs["rag_assistant"]["system_msg"],
        )

    def _execute_function_calling(
        self,
        chat_messages: list[dict[str, str]],
        max_token_number: int,
        encoding_model: str,
    ):
        num_tokens_in_messages_before = count_tokens_of_conversation(
            chat_messages=chat_messages, encoding_model=encoding_model
        )
        if num_tokens_in_messages_before > max_token_number:
            raise TokenLengthExceedsMaxTokenNumber
        response = self.openai_completion_call(
            chat_messages=chat_messages, tools=self.function_definitions
        )
        curr_response_message = response.choices[0].message
        tool_calls = deepcopy(curr_response_message.tool_calls)
        final_answer = curr_response_message.content
        agent_answer_data = AgentAnswerData(query_msg_idx=len(chat_messages) - 1)
        # Check if the model wanted to call a function
        if tool_calls:
            while len(tool_calls) > 0:
                print("TOOL_CALL")
                chat_messages.append(curr_response_message)
                self._add_function_call_information(
                    chat_messages=chat_messages,
                    tool_calls=tool_calls,
                    agent_answer_data=agent_answer_data,
                )

                second_response = self.openai_completion_call(
                    chat_messages=chat_messages, tools=self.function_definitions
                )
                second_response_message = second_response.choices[0].message
                final_answer = second_response_message.content
                new_tool_calls = deepcopy(second_response_message.tool_calls)
                if new_tool_calls:
                    # If the model wants to call new functions add them to the tool_calls
                    chat_messages.append(second_response_message)
                    tool_calls.extend(new_tool_calls)
        agent_answer_data.final_answer = final_answer
        return agent_answer_data

    def _add_function_call_information(
        self,
        chat_messages: list[dict[str, str]],
        tool_calls: list[Any],
        agent_answer_data: AgentAnswerData,
    ):
        while len(tool_calls) > 0:
            tool_call = tool_calls.pop(0)
            function_name = tool_call.function.name
            function_to_call = self.available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            function_response += "\n\nIf the responses of the previously made tool calls are enough to answer the question, return the final answer. If the responses of the tool calls contain not enough information, then call one appropriate function."
            agent_answer_data.add_function_response(function_response)
            chat_messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

    def get_meta_data(self):
        all_meta_data = []
        for function_name in self.available_functions:
            if hasattr(self.available_functions[function_name], "meta_data"):
                meta_data = self.available_functions[function_name].meta_data
                all_meta_data.append(
                    {"function": function_name, "meta_data": meta_data}
                )
                self.available_functions[function_name].meta_data = []
        return all_meta_data

    def run(self, query: str, chat_messages: list[dict[str, str]]) -> AgentAnswerData:
        if not chat_messages or len(chat_messages) == 0:
            chat_messages = self.insert_initial_system_msg(chat_messages=chat_messages)
        chat_messages.append({"role": "user", "content": query})
        agent_answer = self._execute_function_calling(
            chat_messages=chat_messages,
            max_token_number=get_max_token_number(model_name=self.model_name),
            encoding_model=tiktoken.get_encoding("cl100k_base"),
        )
        return agent_answer
