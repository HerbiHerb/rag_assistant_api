from typing import List, Dict


def insert_initial_system_msg(initial_system_msg: str, chat_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
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
        {"role": chat_message["role"], "content": chat_message["content"]} for chat_message in chat_messages
    ]
    return new_chat_messages
