from abc import ABC, abstractmethod
from typing import Callable
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
