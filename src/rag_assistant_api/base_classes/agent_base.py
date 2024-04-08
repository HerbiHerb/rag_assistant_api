from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from openai import OpenAI
from openai._types import NOT_GIVEN
from langchain.chains.llm import LLMChain


class AgentBase(ABC, BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def run(query: str) -> str:
        pass


class OpenAIAgent(AgentBase):
    openai_client: OpenAI
    model_name: str
    initial_system_msg: str

    def insert_initial_system_msg(
        self, chat_messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Generates the initial chat messages for the openai llms.

        Returns:
            A list containing the initial chat message
        """
        chat_messages = [] if not chat_messages else chat_messages
        chat_messages.append({"role": "system", "content": self.initial_system_msg})
        return chat_messages

    def openai_completion_call(
        self,
        chat_messages: list[dict],
        tools: list[str] = NOT_GIVEN,
        tool_choice: str = "auto",
    ):
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
