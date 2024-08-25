from ..base_classes.agent_base import AgentBase
from ..agents.langchain.langchain_agents.langchain_openai_agent import (
    LangchainOpenAIAgent,
)
from ..agents.openai.openai_agents.azure_openai_assistant import AzureOpenAIAssistant


class AgentFactory:
    factories = {}

    @staticmethod
    def create_agent(
        config_data: dict[dict[str, str]],
        # user_id: int,
        # document_filter: dict[str, list[str]] = None,
        **kwargs
    ) -> AgentBase:
        agent_type = config_data["usage_settings"]["agent_type"]
        if not agent_type in AgentFactory.factories:
            try:
                AgentFactory.factories[agent_type] = eval(agent_type + ".Factory()")
            except NameError as e:
                raise NameError(
                    "NameError: Please define one of the following agent types in the config.yaml file for agent_type: LangchainOpenAIAgent, OpenAIFunctionsAgent, AzureOpenAIAssistant"
                )
        return AgentFactory.factories[agent_type].initialize_agent(**kwargs)
