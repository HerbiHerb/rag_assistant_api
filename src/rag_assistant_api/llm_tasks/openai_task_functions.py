import os
import openai
import yaml
from ..utils.data_processing_utils import list_str_conversion


def summarize_text(text: str):
    with open(os.getenv("CONFIG_FP"), "r") as file:
        config_data = yaml.safe_load(file)
    with open(os.getenv("PROMPT_CONFIGS_FP"), "r") as file:
        prompt_configs = yaml.safe_load(file)
    chat_messages = [
        {"role": "system", "content": prompt_configs["generate_summary"]["system_msg"]},
        {"role": "user", "content": text},
    ]
    response = openai.chat.completions.create(
        messages=chat_messages,
        model=config_data["language_models"]["model_name"],
        temperature=config_data["language_models"]["temp"],
    )

    return response


def reformulate_query(query: str) -> list[str]:
    with open(os.getenv("CONFIG_FP"), "r") as file:
        config_data = yaml.safe_load(file)
    with open(os.getenv("PROMPT_CONFIGS_FP"), "r") as file:
        prompt_configs = yaml.safe_load(file)
    chat_messages = [
        {
            "role": "system",
            "content": prompt_configs["reformulate_query"]["system_msg"],
        },
        {"role": "user", "content": query},
    ]
    response = openai.chat.completions.create(
        messages=chat_messages,
        model="gpt-3.5-turbo-0125",
        temperature=config_data["language_models"]["temp"],
    )
    converted_json_list = list_str_conversion(response.choices[0].message.content)
    return converted_json_list if converted_json_list else [response]
