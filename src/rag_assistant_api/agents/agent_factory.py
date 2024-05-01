import yaml
import os
from ..base_classes.agent_base import AgentBase
from ..agents.openai.openai_agents.openai_functions_agent import OpenAIFunctionsAgent
from ..utils.file_loading import load_yaml_file


class AgentFactory:
    factories = {}

    @staticmethod
    def create_agent(document_filter: dict = None) -> AgentBase:
        config_data = load_yaml_file(yaml_file_fp=os.getenv("CONFIG_FP"))
        agent_type = config_data["usage_settings"]["agent_type"]
        if not agent_type in AgentFactory.factories:
            AgentFactory.factories[agent_type] = eval(agent_type + ".Factory()")
        return AgentFactory.factories[agent_type].initialize_agent(document_filter)
