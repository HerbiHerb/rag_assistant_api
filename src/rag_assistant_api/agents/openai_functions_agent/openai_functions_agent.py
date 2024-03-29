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
from ...base_classes.agent_base import AgentBase

# from ..agent_tools.tools import DocumentSearch
from ...data_processing.config_schemas import PineconeConfig, DataProcessingConfig
from ...data_processing.pinecone.pinecone_database_handler import (
    PineconeDatabaseHandler,
)


class AgentData(BaseModel):
    openai_key: str
    # model_name: str
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
    function_responses: List[str] = Field(default=[])

    def add_function_response(self, new_response: str) -> None:
        self.function_responses.append(new_response)


class TokenLengthExceedsMaxTokenNumber(Exception):
    """The token number exceeds the maximum token number the model can handle."""


class ModelNotIncluded(Exception):
    """The model name is not supported"""


def get_max_token_number(model_name: str) -> int:
    token_num_lookup = {
        "gpt-3.5-turbo-0125": 16000,
        "gpt-3.5-turbo-1106": 16000,
        "gpt-4-1106-preview": 128000,
    }
    if model_name not in token_num_lookup:
        raise ModelNotIncluded
    return token_num_lookup[model_name]


def count_tokens(text: str, encoding_model: tiktoken.Encoding):
    # encoding_model = tiktoken.get_encoding(encoding_model)
    num_tokens = len(encoding_model.encode(text))
    return num_tokens


def count_tokens_of_conversation(
    chat_messages: List[Dict[str, str]], encoding_model: str
):
    num_tokens = 0
    for message in chat_messages:
        if isinstance(message, dict):
            text = message["content"]
            num_tokens += count_tokens(text=text, encoding_model=encoding_model)
    return num_tokens


def handle_tool_call(
    openai_client: OpenAI,
    model_name: str,
    chat_messages: List[Dict[str, str]],
    tool_call: List[Any],
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
    second_response = openai_client.chat.completions.create(
        model=model_name,
        messages=chat_messages,
        tools=TOOLS_LIST,
        tool_choice="auto",
    )
    second_response_message = second_response.choices[0].message
    final_answer = second_response_message.content
    tool_calls = deepcopy(second_response_message.tool_calls)
    return final_answer, tool_calls, second_response_message


def generate_answer(
    openai_client: OpenAI,
    chat_messages: List[Dict[str, str]],
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
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=chat_messages,
        tools=TOOLS_LIST,
        tool_choice="auto",
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
            final_answer, new_tool_calls, curr_response_message = handle_tool_call(
                openai_client=openai_client,
                model_name=model_name,
                chat_messages=chat_messages,
                tool_call=tool_call,
                agent_answer_data=agent_answer_data,
                available_functions=available_functions,
            )
            if new_tool_calls:
                tool_calls.extend(new_tool_calls)
    agent_answer_data.final_answer = final_answer
    return agent_answer_data


def initialize_openai_functions_agent(model_name: str, embedding_model: str):
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
    available_functions = {
        "fetch_relevant_information": PineconeDocumentSearch(
            embedding_model=embedding_model, database_handler=database_handler
        ),
    }
    rag_agent = OpenAIFunctionsAgent(
        openai_client=OpenAI(),
        model_name=model_name,
        available_functions=available_functions,
    )
    return rag_agent


class OpenAIFunctionsAgent(AgentBase, BaseModel):
    openai_client: OpenAI
    model_name: str
    available_functions: dict

    class Config:
        arbitrary_types_allowed = True

    def _read_and_reset_meta_data(self):
        for function_name in self.available_functions:
            if hasattr(self.available_functions[function_name], "meta_data"):
                meta_data = self.available_functions[function_name].meta_data

    def run(self, query: str, chat_messages: List[Dict[str, str]]) -> AgentAnswerData:

        chat_messages.append({"role": "user", "content": query})
        agent_answer = generate_answer(
            openai_client=self.openai_client,
            chat_messages=deepcopy(chat_messages),
            model_name=self.model_name,
            max_token_number=get_max_token_number(model_name=self.model_name),
            encoding_model=tiktoken.get_encoding("cl100k_base"),
            available_functions=self.available_functions,
        )
        return agent_answer
