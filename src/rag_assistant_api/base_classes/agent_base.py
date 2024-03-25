from abc import ABC, abstractmethod
from typing import Callable
from pydantic import BaseModel, Field
from langchain_openai import AzureOpenAI
from langchain.chains.llm import LLMChain


class AgentBase(ABC):
    @abstractmethod
    def run(query: str) -> str:
        pass
