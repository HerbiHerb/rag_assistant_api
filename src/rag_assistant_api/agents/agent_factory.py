from ..base_classes.agent_base import AgentBase
from ..agents.openai.openai_agents.openai_functions_agent import OpenAIFunctionsAgent
from ..agents.langchain.langchain_agents.langchain_openai_agent import (
    LangchainOpenAIAgent,
)


class AgentFactory:
    factories = {}

    @staticmethod
    def create_agent(
        config_data: dict[dict[str, str]], document_filter: dict = None
    ) -> AgentBase:
        agent_type = config_data["usage_settings"]["agent_type"]
        if not agent_type in AgentFactory.factories:
            try:
                AgentFactory.factories[agent_type] = eval(agent_type + ".Factory()")
            except NameError as e:
                raise NameError(
                    "NameError: Please define one of the following agent types in the config.yaml file: OpenAIFunctionsAgent"
                )
        return AgentFactory.factories[agent_type].initialize_agent(document_filter)
