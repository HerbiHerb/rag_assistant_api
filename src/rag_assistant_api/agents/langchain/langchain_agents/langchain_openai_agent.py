import os
from typing import Any, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from ....base_classes.agent_base import LangchainAgent
from ....vector_database.vector_db_factory import VectorDBFactory
from ....llm_functionalities.embedding_models.embedding_model_factory import (
    EmbeddingModelFactory,
)
from ....utils.file_loading import load_yaml_file
from ..langchain_tools.tools import DocumentSearch, SQLQuerySearch
from ....data_structures.data_structures import AgentAnswerData


class LangchainOpenAIAgent(LangchainAgent):
    functions: list[BaseTool]
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
                llm_service=config_data["usage_settings"]["llm_service"],
                embedding_model_name=config_data["language_models"]["embedding_model"],
            )
            functions = [
                DocumentSearch(
                    embedding_model=embedding_model,
                    database_handler=database_handler,
                ),
                SQLQuerySearch(
                    embedding_model=embedding_model,
                    database_handler=database_handler,
                ),
            ]
            function_definitions = [
                convert_to_openai_function(func) for func in functions
            ]
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        prompt_configs[
                            config_data["usage_settings"]["agent_prompt_config"]
                        ]["system_msg"],
                    ),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )

            if config_data["usage_settings"]["llm_service"] == "azure":
                llm = AzureChatOpenAI(
                    openai_api_version=os.getenv("OPENAI_API_VERSION"),
                    azure_deployment=config_data["language_models"]["model_name"],
                    temperature=0,
                )
            elif config_data["usage_settings"]["llm_service"] == "openai":
                llm = ChatOpenAI(
                    model=config_data["language_models"]["model_name"], temperature=0
                )
            else:
                raise ValueError(
                    "If you use LangchainOpenAIAgent the llm_service variable should be either openai or azure!!"
                )

            agent = create_openai_functions_agent(
                llm,
                functions,
                prompt,
            )
            agent_executor = AgentExecutor(
                agent=agent,
                tools=functions,
                verbose=True,
                return_intermediate_steps=True,
            )
            return LangchainOpenAIAgent(
                model=agent_executor,
                functions=functions,
                function_definitions=function_definitions,
                initial_system_msg=prompt_configs[
                    config_data["usage_settings"]["agent_prompt_config"]
                ]["system_msg"],
            )

    def _convert_chat_messages(
        self, chat_messages: list[dict[str, str]]
    ) -> list[Union[SystemMessage, HumanMessage, AIMessage]]:
        """
        Converts a dict of chat messages into the corresponding langchain data types.

        Args:
            chat_messages (list[dict[str, str]]): The list of the chat messages.

        Returns:
            converted_chat_messages: The list of converted chat messages.
        """
        converted_chat_messages = []
        for msg in chat_messages:
            if msg["role"] == "system":
                converted_chat_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                converted_chat_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                converted_chat_messages.append(AIMessage(content=msg["content"]))
        return converted_chat_messages

    def get_meta_data(
        self, intermediate_steps: list[tuple[Any, Any]]
    ) -> list[dict[str, str]]:
        """
        Extracts the meta data of the function call containing the ground truth information

        Args:
            intermediate_steps (list[tuple[Any, Any]]): The list of intermediate steps of a function call

        Returns:
            combined_response_data (list[str]): The list of the meta data used by the model to answer the user query
        """
        combined_response_data = []
        for step in intermediate_steps:
            function_response_data = step[1]
            combined_response_data.append(function_response_data)
        return combined_response_data

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
        converted_chat_messages = self._convert_chat_messages(
            chat_messages=chat_messages
        )
        response = self.model.invoke(
            {
                "input": query,
                "chat_history": converted_chat_messages,
            }
        )
        chat_messages.append({"role": "user", "content": query})
        agent_answer_data = AgentAnswerData(
            query_msg_idx=len(chat_messages) - 1,
            final_answer=response["output"],
            chat_messages=chat_messages,
            function_responses=self.get_meta_data(response["intermediate_steps"]),
        )
        return agent_answer_data
