import os
import yaml
import openai
import pinecone


def set_openai_credentials():
    with open(os.environ["CREDENTIALS_FP"], "r") as credential_config:
        dict_of_credentials = yaml.safe_load(credential_config)
    os.environ["OPENAI_API_KEY"] = dict_of_credentials["OPENAI_API_KEY"]
    # os.environ["TA_API_KEY"] = dict_of_credentials["text_analytics_key"]
    # os.environ["TA_ENDPOINT"] = dict_of_credentials["text_analytics_endpoint"]
    openai.api_key = os.environ["OPENAI_API_KEY"]
    pinecone.init(
        api_key=dict_of_credentials["pinecone_api_key"],
        environment=dict_of_credentials["pinecone_environment"],
    )


def set_api_credentials():
    with open(
        os.environ["CREDENTIALS_FP"],
        "r",
    ) as credential_config:
        dict_of_credentials = yaml.safe_load(credential_config)

    for key, value in zip(dict_of_credentials.keys(), dict_of_credentials.values()):
        os.environ[key] = str(value)
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
        pinecone.init(
            api_key=dict_of_credentials["pinecone_api_key"],
            environment=dict_of_credentials["pinecone_environment"],
        )
    except Exception as e:
        print(e)
