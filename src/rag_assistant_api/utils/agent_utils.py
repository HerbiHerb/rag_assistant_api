from typing import List, Dict
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import tiktoken
from ..agents.exceptions import ModelNotIncluded


def get_max_token_number(model_name: str) -> int:
    token_num_lookup = {
        "gpt-3.5-turbo-0125": 16000,
        "gpt-3.5-turbo-1106": 16000,
        "gpt-4-1106-preview": 128000,
    }
    if model_name not in token_num_lookup:
        raise ModelNotIncluded
    return token_num_lookup[model_name]


def count_tokens(text: str, encoding_model: tiktoken.Encoding):
    # encoding_model = tiktoken.get_encoding(encoding_model)
    num_tokens = len(encoding_model.encode(text))
    return num_tokens


def count_tokens_of_conversation(
    chat_messages: List[Dict[str, str]], encoding_model: str
):
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


def extract_openai_chat_messages(chat_messages: List[Dict[str, str]]):
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
            role = chat_message.role
            for tool_call in chat_message.tool_calls:
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
                converted_chat_messages.append(
                    {
                        "role": role,
                        "content": f"$FUNCTION_CALL\n\nPreviously made function call\n\nFunction name: {function_name}\narguments: {arguments}",
                    }
                )
        elif chat_message["role"] == "tool":
            continue
        else:
            converted_chat_messages.append(chat_message)
    return converted_chat_messages
