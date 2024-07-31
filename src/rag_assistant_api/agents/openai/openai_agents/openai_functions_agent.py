from openai import OpenAI
from pydantic import BaseModel
from typing import Any
import json
import os
import pinecone
import yaml
from copy import deepcopy
import tiktoken
from ..openai_functions.function_definitions import (
    PineconeDocumentSearch,
    PineconeDocumentFilterSearch,
    DocumentAnalyzer,
    DOCUMENT_SEARCH,
    DOCUMENT_FILTER_SEARCH,
    DOCUMENT_ANALYZER,
)
from ....base_classes.agent_base import OpenAIAgent
from ....data_structures.data_structures import AgentAnswerData
from ...exceptions import TokenLengthExceedsMaxTokenNumber
from ....utils.agent_utils import count_tokens_of_conversation, get_max_token_number
from ....vector_database.vector_db_factory import VectorDBFactory
from ....llm_functionalities.embedding_models.embedding_model_factory import (
    EmbeddingModelFactory,
)
from ....utils.file_loading import load_yaml_file


class OpenAIFunctionsAgent(OpenAIAgent):
    available_functions: dict[str, BaseModel]
    function_definitions: list[dict]

    class Factory:
        """
        The factory class to initialize the agent based on the definition in the config.yaml file.
        """

        def initialize_agent(self, document_filter: dict = None):
            config_data = load_yaml_file(yaml_file_fp=os.getenv("CONFIG_FP"))
            prompt_configs = load_yaml_file(yaml_file_fp=os.getenv("PROMPT_CONFIGS_FP"))
            database_handler = VectorDBFactory.create_vector_db_instance(
                vector_db_cls=config_data["usage_settings"]["vector_db"],
                config_data=config_data,
            )
            embedding_model = EmbeddingModelFactory.create_embedding_model(
                embedding_model_cls=config_data["usage_settings"][
                    "embeddding_model_cls"
                ],
                embedding_model_name=config_data["language_models"]["embedding_model"],
            )
            available_functions = {
                "document_search": PineconeDocumentSearch(
                    embedding_model=embedding_model,
                    database_handler=database_handler,
                ),
                "document_filter_search": PineconeDocumentFilterSearch(
                    embedding_model=embedding_model,
                    database_handler=database_handler,
                    filter_dict=document_filter,
                ),
                "document_analyzer": DocumentAnalyzer(
                    embedding_model=embedding_model,
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
                initial_system_msg=prompt_configs[
                    config_data["usage_settings"]["agent_prompt_config"]
                ]["system_msg"],
            )

    def _execute_function_calling(
        self,
        chat_messages: list[dict[str, str]],
        max_token_number: int,
        encoding_model: str,
    ) -> AgentAnswerData:
        """
        The function which handles the function calling steps of the agent. The agent decides on his own which function to take to answer the user query.

        Args:
            chat_messages (list[dict[str, str]]): The list of past chat messages from the current conversation (chat history)
            max_token_number (int): The maximum number of tokens the agent can handle.
            encoding_model (str): The model used to count the token number.

        Returns:
            AgentAnswerData: Data object containing the answer from the agent, the chat history and the function responses
        """
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
                chat_messages.append(curr_response_message)
                self._add_function_call_information(
                    chat_messages=chat_messages,
                    tool_calls=tool_calls,
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
    ):
        """
        This function adds the function calling information to the chat messages.

        Args:
            chat_messages (list[dict[str, str]]): The list of past chat messages from the current conversation (chat history)
            tool_calls (list[Any]): The list of current tool calls to fetch relevant information.
        Returns:
            None
        """
        while len(tool_calls) > 0:
            tool_call = tool_calls.pop(0)
            print(f"TOOL_CALL: {tool_call.function.name}")
            function_name = tool_call.function.name
            function_to_call = self.available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            function_response += "\n\nIf the responses of the previously made tool calls are enough to answer the question, return the final answer. If the responses of the tool calls contain not enough information, then call one appropriate function."
            chat_messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

    def get_meta_data(self) -> list[dict[str, str]]:
        """
        Extracts the meta data of the function call containing the ground truth information

        Args:
            intermediate_steps (list[tuple[Any, Any]]): The list of intermediate steps of a function call

        Returns:
            combined_response_data (list[str]): The list of the meta data used by the model to answer the user query
        """
        all_meta_data = []
        for function_name in self.available_functions:
            if hasattr(self.available_functions[function_name], "meta_data"):
                meta_data = self.available_functions[function_name].meta_data
                all_meta_data.append(
                    {"function": function_name, "meta_data": meta_data}
                )
                self.available_functions[function_name].meta_data = []
        return all_meta_data

    def run(
        self, query: str, chat_messages: list[dict[str, str]], **kwargs
    ) -> AgentAnswerData:
        """
        The run function to answer the user query with the corresponding chat history.

        Args:
            query (str): The new user query
            chat_messages (list[dict[str, str]]): The list of past chat messages from the current conversation (chat history)

        Returns:
            AgentAnswerData: Data object containing the answer from the agent, the chat history and the function responses
        """
        if not chat_messages or len(chat_messages) == 0:
            chat_messages = self.insert_initial_system_msg(chat_messages=chat_messages)
        chat_messages.append({"role": "user", "content": query})
        # TODO: Replace the hard coded encoding model name and put it in the config file
        agent_answer = self._execute_function_calling(
            chat_messages=chat_messages,
            max_token_number=get_max_token_number(model_name=self.model_name),
            encoding_model=tiktoken.get_encoding("cl100k_base"),
        )
        agent_answer.chat_messages = chat_messages
        agent_answer.function_responses = self.get_meta_data()
        return agent_answer
