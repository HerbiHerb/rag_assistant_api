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
from ..openai_functions.function_definitions import TOOLS_LIST, PineconeDocumentSearch
from ...base_classes.agent_base import OpenAIAgent
from ...vector_database.config_schemas import PineconeConfig, DataProcessingConfig
from ...vector_database.pinecone.pinecone_database_handler import (
    PineconeDatabaseHandler,
)
from ..data_structure import AgentAnswerData
from ..exceptions import TokenLengthExceedsMaxTokenNumber
from ..agent_utils import count_tokens_of_conversation, get_max_token_number


class OpenAIFunctionsAgent(OpenAIAgent):
    available_functions: dict

    @classmethod
    def initialize_openai_functions_agent(cls, model_name: str, embedding_model: str):
        with open(os.getenv("CONFIG_FP"), "r") as file:
            config_data = yaml.safe_load(file)
        data_processing_config = DataProcessingConfig(**config_data["data_processing"])
        pinecone_config = PineconeConfig(**config_data["pinecone_db"])
        database_handler = PineconeDatabaseHandler(
            index=pinecone.Index(pinecone_config.index_name),
            data_processing_config=data_processing_config,
            pinecone_config=PineconeConfig(**config_data["pinecone_db"]),
        )
        available_functions = {
            "fetch_relevant_information": PineconeDocumentSearch(
                embedding_model=OpenAIEmbeddings(model=embedding_model),
                database_handler=database_handler,
            ),
        }
        return OpenAIFunctionsAgent(
            openai_client=OpenAI(),
            model_name=model_name,
            available_functions=available_functions,
        )

    def _execute_function_calling(
        self,
        openai_client: OpenAI,
        chat_messages: list[dict[str, str]],
        model_name: str,
        max_token_number: int,
        encoding_model: str,
        available_functions: dict[callable],
    ):
        num_tokens_in_messages_before = count_tokens_of_conversation(
            chat_messages=chat_messages, encoding_model=encoding_model
        )
        if num_tokens_in_messages_before > max_token_number:
            raise TokenLengthExceedsMaxTokenNumber
        response = self.openai_completion_call(
            chat_messages=chat_messages, tools=TOOLS_LIST
        )
        curr_response_message = response.choices[0].message
        tool_calls = deepcopy(curr_response_message.tool_calls)
        final_answer = curr_response_message.content
        agent_answer_data = AgentAnswerData(query_msg_idx=len(chat_messages) - 1)
        # Step 2: check if the model wanted to call a function
        if tool_calls:
            while len(tool_calls) > 0:
                print("TOOL_CALL")
                chat_messages.append(curr_response_message)
                tool_call = tool_calls.pop(0)
                final_answer, new_tool_calls, curr_response_message = (
                    self._handle_tool_call(
                        chat_messages=chat_messages,
                        tool_call=tool_call,
                        agent_answer_data=agent_answer_data,
                        available_functions=available_functions,
                    )
                )
                if new_tool_calls:
                    tool_calls.extend(new_tool_calls)
        agent_answer_data.final_answer = final_answer
        return agent_answer_data

    def _handle_tool_call(
        self,
        chat_messages: list[dict[str, str]],
        tool_call: list[Any],
        agent_answer_data: AgentAnswerData,
        available_functions: dict[callable],
    ):
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
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
        second_response = self.openai_completion_call(
            chat_messages=chat_messages, tools=TOOLS_LIST
        )
        second_response_message = second_response.choices[0].message
        final_answer = second_response_message.content
        tool_calls = deepcopy(second_response_message.tool_calls)
        return final_answer, tool_calls, second_response_message

    def _read_and_reset_meta_data(self):
        for function_name in self.available_functions:
            if hasattr(self.available_functions[function_name], "meta_data"):
                meta_data = self.available_functions[function_name].meta_data

    def run(self, chat_messages: list[dict[str, str]]) -> AgentAnswerData:
        agent_answer = self._execute_function_calling(
            openai_client=self.openai_client,
            chat_messages=chat_messages,
            model_name=self.model_name,
            max_token_number=get_max_token_number(model_name=self.model_name),
            encoding_model=tiktoken.get_encoding("cl100k_base"),
            available_functions=self.available_functions,
        )
        return agent_answer
