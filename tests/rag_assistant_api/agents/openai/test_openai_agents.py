import os
import openai
from pydantic.types import confloat
import yaml
from dotenv import load_dotenv
import pytest
from rag_assistant_api.data_structures.data_structures import AgentAnswerData
from rag_assistant_api.base_classes.agent_base import AgentBase
from rag_assistant_api.agents.agent_factory import AgentFactory


def test_openai_agent_type(config_data):
    load_dotenv()
    rag_model = AgentFactory.create_agent(config_data=config_data)
    assert isinstance(rag_model, AgentBase)


def test_openai_agent_with_empty_chat_messages():
    load_dotenv()
    with open(
        os.environ["CONFIG_FP"],
        "r",
    ) as file:
        config_data = yaml.safe_load(file)
    if config_data["usage_settings"]["llm_service"] == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
    rag_model = AgentFactory.create_agent(config_data=config_data)
    query = "Kannst du mir helfen?"
    chat_messages = []
    response = rag_model.run(query=query, chat_messages=chat_messages)
    assert isinstance(response, AgentAnswerData)
    assert len(response.chat_messages) > 0
    assert len(response.final_answer) > 0


def test_openai_agent_with_chat_messages():
    load_dotenv()
    with open(
        os.environ["CONFIG_FP"],
        "r",
    ) as file:
        config_data = yaml.safe_load(file)
    if config_data["usage_settings"]["llm_service"] == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
    rag_model = AgentFactory.create_agent(config_data=config_data)
    query = "Kannst du mir helfen?"
    chat_messages = [
        {
            "role": "system",
            "content": "Du bist ein Assistent, der dem Nutzer Fragen beantwortet.",
        }
    ]
    response = rag_model.run(query=query, chat_messages=chat_messages)
    assert isinstance(response, AgentAnswerData)
    assert len(response.chat_messages) > 0
    assert len(response.final_answer) > 0


def test_wrong_agent_config(config_data):
    config_data["usage_settings"]["agent_type"] = "WrongAgent"
    with pytest.raises(NameError) as excinfo:
        rag_model = AgentFactory.create_agent(config_data=config_data)
    assert (
        str(excinfo.value)
        == "NameError: Please define one of the following agent types in the config.yaml file: OpenAIFunctionsAgent"
    )
