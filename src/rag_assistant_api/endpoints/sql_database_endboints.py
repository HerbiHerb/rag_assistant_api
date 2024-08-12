import os
import yaml
import json
import openai
from flask import jsonify, request
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv
from ..vector_database.vector_db_factory import VectorDBFactory
from ..init_flask_app import app
from ..local_database.database_models import Conversation, User, Document, SpeechQuery
from ..utils.file_loading import load_yaml_file


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
    conv_id = Conversation.get_latest_conversation_id(user_id=user_id)
    return jsonify({"conv_id": conv_id})


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
    if not conv_id:
        return jsonify(
            {
                "chat_messages": [],
                "sources": [],
            }
        )
    try:
        chat_messages = Conversation.get_chat_messages(conv_id=conv_id)
        sources = Conversation.get_meta_data_for_chat_messages(conv_id=conv_id)
        return jsonify(
            {
                "chat_messages": chat_messages,
                "sources": sources,
            }
        )
    except Exception as e:
        print(e)
        return f"An error occured {e}"


@app.route("/get_all_doument_meta_data", methods=["POST"])
def get_all_doument_meta_data():
    """
    Retrieves all document meta_data for a given user ID.

    Returns:
        JSON response containing all document meta data.
    """
    request_data = json.loads(request.data)
    user_id = request_data["user_id"]
    documents = Document.get_all_documents_from_user(user_id=user_id)
    if not documents:
        config_data = load_yaml_file(yaml_file_fp=os.getenv("CONFIG_FP"))
        database_handler = VectorDBFactory.create_vector_db_instance(
            vector_db_cls=config_data["usage_settings"]["vector_db"],
            config_data=config_data,
        )
        metadata = database_handler.get_all_document_meta_data()
        for entry in metadata:
            Document.save_document(
                user_id=user_id,
                document_name=entry["document_name"],
                document_type=entry["document_type"],
            )
        return jsonify(metadata)
    return jsonify(documents)


@app.route("/add_new_speech_query", methods=["POST"])
def add_new_speech_query():
    """
    Add new spoken query comming from the user

    Returns:
        A status message
    """
    response = {"success": False}
    request_data = json.loads(request.data)
    user_id = request_data["user_id"]
    query = request_data["speech_query"]
    speech_query_id = SpeechQuery.save_user_query(user_id=user_id, query=query)
    response["speech_query_id"] = speech_query_id
    response["success"] = True
    return jsonify(response)
