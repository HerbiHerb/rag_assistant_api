import pytest

from src.rag_assistant_api.modules.qa_agent.agent_config_schema import (
    QuestionAnsweringAgentConfig,
)


@pytest.fixture
def agent_config_davinci3():
    return {
        "agent_executor": {"selector": "LLMSingleActionAgent"},
        "language_model": {
            "service": "OpenAI",
            "model_name": "text-davinci-003",
            "temp": 0.0,
        },
        "vector_db": {
            "db_source_path": None,
            "db_type": "chroma",
        },
    }


@pytest.fixture
def agent_config_davinci3_with_azure():
    return {
        "agent_executor": {"selector": "ZeroShotReactAgent"},
        "language_model": {
            "service": "AzureOpenAI",
            "model_name": "text-davinci-003",
            "deployment_name": "davinci_model",
            "temp": 0.0,
        },
        "vector_db": {
            "db_source_path": None,
            "db_type": "chroma",
        },
    }


@pytest.fixture
def agent_config_gpt_4():
    return {
        "agent_executor": {"selector": "LLMSingleActionAgent"},
        "language_model": {
            "service": "OpenAI",
            "model_name": "gpt-4",
            "temp": 0.0,
        },
        "vector_db": {
            "db_source_path": None,
            "db_type": "chroma",
        },
    }


@pytest.fixture
def agent_config_text_curie_001():
    return {
        "agent_executor": {"selector": "LLMSingleActionAgent"},
        "language_model": {
            "service": "OpenAI",
            "model_name": "text-curie-001",
            "temp": 0.0,
        },
        "vector_db": {
            "db_source_path": None,
            "db_type": "chroma",
        },
    }


@pytest.fixture
def default_agent_config_davinci3(agent_config_davinci3):
    return QuestionAnsweringAgentConfig(**agent_config_davinci3)


@pytest.fixture
def default_agent_config_gpt4(agent_config_gpt_4):
    return QuestionAnsweringAgentConfig(**agent_config_gpt_4)


@pytest.fixture
def default_agent_text_curie_001(agent_config_text_curie_001):
    return QuestionAnsweringAgentConfig(**agent_config_text_curie_001)


@pytest.fixture
def defualt_agent_config_davinci3_with_azure(agent_config_davinci3_with_azure):
    return QuestionAnsweringAgentConfig(**agent_config_davinci3_with_azure)
