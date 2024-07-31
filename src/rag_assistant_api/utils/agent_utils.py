import os
from typing import List, Dict
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import tiktoken
from ..agents.exceptions import ModelNotIncluded
from ..utils.file_loading import load_yaml_file


def get_max_token_number(model_name: str) -> int:
    """
    Function to get the maximum token number of a specific model.

    Args:
        model_name (str): The the name mof the model

    Returns:
        int: The maximum token number the model can handle
    """
    # TODO: Put these hard coded lookups in a config file
    token_num_lookup = {
        "gpt-3.5-turbo-0125": 16000,
        "gpt-3.5-turbo-1106": 16000,
        "gpt-4-1106-preview": 128000,
        "gpt-4-turbo": 128000,
    }
    if model_name not in token_num_lookup:
        raise ModelNotIncluded
    return token_num_lookup[model_name]


def count_tokens(text: str, encoding_model: tiktoken.Encoding):
    return len(encoding_model.encode(text))


def count_tokens_of_conversation(
    chat_messages: list[dict[str, str]], encoding_model: str
):
    """
    Function to count the tokens of the current conversation.

    Args:
        chat_messages (list[dict[str, str]]): The chat messages of the current conversation
        encoding_model (str): The encoding model name to count the tokens

    Returns:
        int: The token number of the conversation
    """
    num_tokens = 0
    for message in chat_messages:
        if isinstance(message, dict):
            text = message["content"]
            num_tokens += count_tokens(text=text, encoding_model=encoding_model)
    return num_tokens


def insert_initial_system_msg(
    initial_system_msg: str, chat_messages: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Generates the initial chat messages for the openai llms.

    Returns:
        A list containing the initial chat message
    """
    chat_messages.append({"role": "system", "content": initial_system_msg})
    return chat_messages


def extract_openai_chat_messages(chat_messages: list[dict[str, str]]):
    """
    Extracts the relevant parts of the chat messages for the llm.

    Returns:
        A list containing the chat messages containing only the role and the content.
    """
    new_chat_messages = [
        {"role": chat_message["role"], "content": chat_message["content"]}
        for chat_message in chat_messages
    ]
    return new_chat_messages


def cleanup_function_call_messages(chat_messages: list[dict[str, str]]):
    converted_chat_messages = []
    for chat_message in chat_messages:
        if isinstance(chat_message, ChatCompletionMessage):
            config_data = load_yaml_file(yaml_file_fp=os.getenv("CONFIG_FP"))
            role = chat_message.role
            for tool_call in chat_message.tool_calls:
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
                converted_chat_messages.append(
                    {
                        "role": role,
                        "content": config_data["usage_settings"]["function_call_prefix"]
                        + f"\n\nPreviously made function call\n\nFunction name: {function_name}\narguments: {arguments}",
                    }
                )
        elif chat_message["role"] == "tool":
            continue
        else:
            converted_chat_messages.append(chat_message)
    return converted_chat_messages
