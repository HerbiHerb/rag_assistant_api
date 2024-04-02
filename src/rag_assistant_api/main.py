import os
import yaml
import json
import openai
from copy import deepcopy
from flask import jsonify, request
import pinecone
from dotenv import load_dotenv
from .init_flask_app import app
from .local_database.database_models import Conversation, User
from .agents.agent_utils import (
    insert_initial_system_msg,
    extract_openai_chat_messages,
    cleanup_function_call_messages,
)
from .agents.prompts import INITIAL_SYSTEM_MSG
from .agents.openai_functions_agent.openai_functions_agent import OpenAIFunctionsAgent
from .vector_database.config_schemas import PineconeConfig, DataProcessingConfig
from .vector_database.pinecone.pinecone_database_handler import PineconeDatabaseHandler
from .vector_database.pinecone.generate_pinecone_db import generate_database


@app.route("/register_new_user", methods=["POST"])
def register_new_user():
    """Checks if username and password are correct"""
    request_data = json.loads(request.data)
    if (
        not request_data
        or "username" not in request_data
        or "password" not in request_data
    ):
        return (
            jsonify({"error": "Request must contain a 'username' and a 'password'"}),
            400,
        )
    username = request_data["username"]
    password = request_data["password"]
    user_exists = User.check_user_exists(username)
    if not user_exists:
        user_id = User.save_new_user(username, password)
        return jsonify({"user_id": user_id})
    else:
        return jsonify({"error": "User already exists"})


@app.route("/check_username_exists", methods=["POST"])
def check_username_exists():
    """Checks if the username already exists"""
    request_data = json.loads(request.data)
    if not request_data or "username" not in request_data:
        return (
            jsonify({"error": "Request must contain a 'username' and a 'password'"}),
            400,
        )
    username = request_data["username"]
    user_exists = User.check_user_exists(username)
    return jsonify({"user_exists": user_exists})


@app.route("/check_username_and_password", methods=["POST"])
def check_username_and_password():
    """Checks if username and password are correct"""
    request_data = json.loads(request.data)
    if (
        not request_data
        or "username" not in request_data
        or "password" not in request_data
    ):
        return (
            jsonify({"error": "Request must contain a 'username' and a 'password'"}),
            400,
        )
    username = request_data["username"]
    password = request_data["password"]
    user_id = User.check_username_and_password(username, password)
    return jsonify({"user_id": user_id})


@app.route("/execute_rag", methods=["POST"])
def execute_rag():
    """
    Handles the conversation with an AI agent. It processes the incoming query,
    retrieves or starts a new conversation, generates a response using the AI model,
    and returns the AI's response along with context information.

    Returns:
        JSON response containing the answer from the AI agent and any source information used.
    """
    request_data = json.loads(request.data)
    if not request_data or "query" not in request_data or "user_id" not in request_data:
        return jsonify({"error": "Request must contain a 'query' key"}), 400

    query = request_data["query"]
    user_id = request_data["user_id"]
    conv_id = Conversation.get_latest_conversation(user_id=user_id)
    if conv_id == None:
        conv_id = Conversation.generate_new_conversation(user_id=user_id)
        if not conv_id:
            return (
                jsonify(
                    {"error": "An error occured when generating a new conversation"}
                ),
                400,
            )
    chat_messages = Conversation.get_chat_messages(conv_id=conv_id)
    if not chat_messages or len(chat_messages) == 0:
        chat_messages = insert_initial_system_msg(
            initial_system_msg=INITIAL_SYSTEM_MSG, chat_messages=chat_messages
        )
    rag_model = OpenAIFunctionsAgent.initialize_openai_functions_agent(
        model_name="gpt-3.5-turbo-0125", embedding_model="text-embedding-ada-002"
    )
    chat_messages = extract_openai_chat_messages(chat_messages=chat_messages)
    chat_messages.append({"role": "user", "content": query})
    agent_answer = rag_model.run(chat_messages=chat_messages)
    chat_messages = cleanup_function_call_messages(chat_messages=chat_messages)
    context_meta = []
    assistant_message = {
        "role": "assistant",
        "content": agent_answer.final_answer,
        "urls": [meta_entry["URL"] for meta_entry in context_meta],
    }
    chat_messages.append(assistant_message)
    Conversation.update_chat_messages(conv_id=conv_id, chat_messages=chat_messages)
    return jsonify({"answer": agent_answer.final_answer, "meta": context_meta})


@app.route("/get_latest_conv_id", methods=["POST"])
def get_latest_conv_id():
    """
    Returns:
    The latest conversation id
    """
    request_data = json.loads(request.data)
    if not request_data or "user_id" not in request_data:
        return jsonify({"error": "Request must contain a 'user_id' key"}), 400
    user_id = request_data["user_id"]
    return jsonify({"conv_id": Conversation.get_latest_conversation(user_id=user_id)})


@app.route("/create_new_conversation", methods=["POST"])
def create_new_conversation():
    """
    Creates a new conversation for a user and returns the conversation ID.

    Returns:
        String indicating the creation of a new conversation with its ID.
    """
    request_data = json.loads(request.data)
    if not request_data or "user_id" not in request_data:
        return jsonify({"error": "Request must contain a 'user_id' key"}), 400
    user_id = request_data["user_id"]
    conv_id = Conversation.generate_new_conversation(user_id=user_id)
    return jsonify({"conv_id": conv_id})


@app.route("/get_chat_messages", methods=["POST"])
def get_chat_messages():
    """
    Retrieves chat messages for a given conversation ID.

    Returns:
        JSON response containing all chat messages of the requested conversation.
        In case of an error, returns an error message.
    """
    request_data = json.loads(request.data)
    if not "query" in request_data:
        return "Request must contain a 'query' key"
    conv_id = request_data["query"]
    try:
        chat_messages = Conversation.get_chat_messages(conv_id=conv_id)
        return jsonify(chat_messages)
    except Exception as e:
        print(e)
        return f"An error occured {e}"


@app.route("/generate_vector_db", methods=["GET"])
def generate_vector_db():
    """
    Generate the vector database.

    Returns:
        JSON response containing the information of the created database.
    """
    with open(os.getenv("VECTOR_DB_CONFIG_FP"), "r") as file:
        config_data = yaml.safe_load(file)
    data_processing_config = DataProcessingConfig(**config_data["data_processing"])
    pinecone_config = PineconeConfig(**config_data["pinecone_db"])
    database_handler = PineconeDatabaseHandler(
        index=pinecone.Index(pinecone_config.index_name),
        data_processing_config=data_processing_config,
        pinecone_config=PineconeConfig(**config_data["pinecone_db"]),
    )
    test = 0
    generate_database(database_handler=database_handler)
    return f"Database generated"


def main():
    """
    Main function to initialize the Flask application.
    It sets up environment variables, loads configuration, and starts the Flask app.
    """
    load_dotenv()
    with open(
        os.environ["CONFIG_FP"],
        "r",
    ) as file:
        config = yaml.safe_load(file)
    if config["language_model"]["service"] == "OpenAI":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
        )
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    main()
