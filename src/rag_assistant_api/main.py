import os
import yaml
import json
import openai
from flask import jsonify, request
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv
from .init_flask_app import app
from .local_database.database_models import Conversation, User, Document
from .utils.agent_utils import (
    extract_openai_chat_messages,
    cleanup_function_call_messages,
)

from .agents.agent_factory import AgentFactory
from .data_structures.data_structures import (
    PineconeConfig,
    ChromaDBConfig,
    DataProcessingConfig,
    DocumentProcessingConfig,
    ConfigFileValidator,
)
from .vector_database.vector_db_factory import VectorDBFactory
from .vector_database.vector_db_utils import (
    generate_database,
    update_database,
)
from .utils.data_processing_utils import (
    extract_meta_data,
    remove_meta_data_from_text,
    split_text_into_parts_and_chapters,
)
from .utils.file_loading import load_yaml_file
from .utils.agent_utils import speak_the_answer


@app.route("/", methods=["GET"])
def home():
    print("Hello world")
    return "Hello world"


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
    try:
        request_data = json.loads(request.data)
        query = request_data["query"]
        user_id = request_data["user_id"]
        conv_id = Conversation.get_latest_conversation_id(user_id=user_id)
        if conv_id == None:
            conv_id = Conversation.generate_new_conversation(user_id=user_id)
        chat_messages = Conversation.get_chat_messages(conv_id=conv_id)
        rag_model = AgentFactory.create_agent(
            config_data=load_yaml_file(yaml_file_fp=os.getenv("CONFIG_FP"))
        )
        chat_messages = extract_openai_chat_messages(chat_messages=chat_messages)
        agent_answer = rag_model.run(
            query=query, chat_messages=chat_messages, conv_id=conv_id
        )
        # speak_the_answer(answer=agent_answer.final_answer)
        chat_messages = cleanup_function_call_messages(
            chat_messages=agent_answer.chat_messages
        )
        chat_messages.append(
            {
                "role": "assistant",
                "content": agent_answer.final_answer,
            }
        )
        Conversation.update_chat_messages(conv_id=conv_id, chat_messages=chat_messages)
        Conversation.save_meta_data(
            conv_id=conv_id,
            msg_idx=len(chat_messages) - 1,
            meta_data=agent_answer.function_responses,
        )

        return jsonify(
            {
                "answer": agent_answer.final_answer,
                "sources": agent_answer.function_responses,
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"A value exception occured! {str(e)}",
        )
    except NotImplementedError as e:
        raise HTTPException(
            status_code=400,
            detail=f"A not implemented exception occured {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"A exception occured! {str(e)}",
        )


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


@app.route("/generate_vector_db", methods=["GET"])
def generate_vector_db():
    """
    Generate the vector database.

    Returns:
        JSON response containing the information of the created database.
    """
    with open(os.getenv("CONFIG_FP"), "r") as file:
        config_data = yaml.safe_load(file)
    database_handler = VectorDBFactory.create_vector_db_instance(
        vector_db_cls=config_data["usage_settings"]["vector_db"],
        config_data=config_data,
    )
    generate_database(database_handler=database_handler)
    return f"Database generated"


@app.route("/upload_document", methods=["POST"])
def upload_document():
    with open(os.getenv("CONFIG_FP"), "r") as file:
        config_data = yaml.safe_load(file)
    database_handler = VectorDBFactory.create_vector_db_instance(
        vector_db_cls=config_data["usage_settings"]["vector_db"],
        config_data=config_data,
    )
    uploaded_text = request.data.decode("utf-8")
    document_config = DocumentProcessingConfig(**config_data["document_processing"])
    meta_data = extract_meta_data(
        extraction_pattern=document_config.meta_data_pattern,
        document_text=uploaded_text,
    )
    uploaded_text = remove_meta_data_from_text(text=uploaded_text)
    update_database(
        text=uploaded_text,
        text_meta_data=meta_data,
        database_handler=database_handler,
        document_processing_config=document_config,
    )
    document_id = Document.save_document(
        user_id=meta_data["user_id"],
        document_type=meta_data["type"],
        document_text=uploaded_text,
    )
    return f"Inserted document id {document_id}"


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
        config_data = yaml.safe_load(file)

    ConfigFileValidator(
        usage_settings=config_data["usage_settings"],
        data_processing_config=DataProcessingConfig(**config_data["data_processing"]),
        document_processing_config=DocumentProcessingConfig(
            **config_data["document_processing"]
        ),
        chroma_db_config=ChromaDBConfig(**config_data["chroma_db"]),
        pinecone_db_config=PineconeConfig(
            api_key=os.getenv("PINECONE_API_KEY"), **config_data["pinecone_db"]
        ),
        prompt_configs_fp=os.getenv("PROMPT_CONFIGS_FP"),
    )
    # Currently only openai and azure is supported
    if config_data["usage_settings"]["llm_service"] in ["openai", "azure"]:
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
