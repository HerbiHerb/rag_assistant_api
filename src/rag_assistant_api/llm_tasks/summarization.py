import os
import openai
import yaml


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
