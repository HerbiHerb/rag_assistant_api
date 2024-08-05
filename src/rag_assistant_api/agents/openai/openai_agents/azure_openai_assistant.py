import os
from typing import Any
import time
from openai import AzureOpenAI
from ....base_classes.agent_base import AgentBase
from ....utils.file_loading import load_yaml_file
from ....data_structures.data_structures import AgentAnswerData
from ....local_database.database_models import Conversation
from ...exceptions import NoAnswerError


class AzureOpenAIAssistant(AgentBase):
    azure_ai_client: AzureOpenAI
    assistant_id: str

    class Factory:
        """
        The factory class to initialize the agent based on the definition in the config.yaml file.
        """

        def initialize_agent(self, document_filter: dict = None):
            config_data = load_yaml_file(yaml_file_fp=os.getenv("CONFIG_FP"))
            azure_ai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )

            assistant = AzureOpenAIAssistant(
                azure_ai_client=azure_ai_client,
                assistant_id=config_data["language_models"]["assistant_id"],
            )
            return assistant

    def get_meta_data() -> list[dict[str, str]]:
        ## TODO: Needs to be implemented
        pass

    def run(
        self, query: str, chat_messages: list[dict[str, str]], conv_id: str
    ) -> AgentAnswerData:
        """
        The run function to answer the user query with the corresponding chat history.

        Args:
            query (str): The new user query
            chat_messages (list[dict[str, str]]): The list of past chat messages from the current conversation (chat history)
            conv_id (str): The id of the current conversation to extract the thread_id. The thread_id is used in the azure assistants api to get the chat history

        Returns:
            AgentAnswerData: Data object containing the answer from the agent, the chat history and the function responses
        """
        thread_id = Conversation.get_thread_id_from_conv_id(conv_id=conv_id)
        # Create a thread
        if not thread_id:
            thread_id = self.azure_ai_client.beta.threads.create().id
            Conversation.update_thread_id_from_conv_id(
                conv_id=conv_id, thread_id=thread_id
            )
        # Add a user question to the thread
        self.azure_ai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=query,
        )
        chat_messages.append({"role": "user", "content": query})
        # Run the thread
        run = self.azure_ai_client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=self.assistant_id
        )

        # Looping until the run completes or fails
        while run.status in ["queued", "in_progress", "cancelling"]:
            time.sleep(1)
            run = self.azure_ai_client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run.id
            )

        if run.status == "completed":
            messages = self.azure_ai_client.beta.threads.messages.list(
                thread_id=thread_id
            )
            if isinstance(messages.data[0].content[0].text, dict):
                answer = messages.data[0].content[0].text["value"]
            else:
                answer = messages.data[0].content[0].text.value
            agent_answer_data = AgentAnswerData(
                query_msg_idx=len(chat_messages) - 1,
                final_answer=answer,
                chat_messages=chat_messages,
            )
            return agent_answer_data
        elif run.status == "requires_action":
            # the assistant requires calling some functions
            # and submit the tool outputs back to the run
            raise NotImplementedError
        else:
            print(run.status)
            raise NoAnswerError
