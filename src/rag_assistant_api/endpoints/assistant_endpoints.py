import os
import yaml
import json
import openai
from flask import jsonify, request
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv
from ..utils.agent_utils import (
    extract_openai_chat_messages,
    cleanup_function_call_messages,
)

from ..agents.agent_factory import AgentFactory
from ..data_structures.data_structures import AgentAnswerData
from ..init_flask_app import app
from ..local_database.database_models import Conversation, SpeechQuery
from ..utils.file_loading import load_yaml_file


def _execute_rag(
    query: str, user_id: int, selected_documents: list[str] = []
) -> AgentAnswerData:
    conv_id = Conversation.get_latest_conversation_id(user_id=user_id)
    if conv_id == None:
        conv_id = Conversation.generate_new_conversation(user_id=user_id)
    chat_messages = Conversation.get_chat_messages(conv_id=conv_id)
    rag_model = AgentFactory.create_agent(
        config_data=load_yaml_file(yaml_file_fp=os.getenv("CONFIG_FP")),
        user_id=user_id,
        document_filter={"document_name": {"$in": list(selected_documents)}},
    )
    chat_messages = extract_openai_chat_messages(chat_messages=chat_messages)
    agent_answer = rag_model.run(
        query=query, chat_messages=chat_messages, conv_id=conv_id
    )
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
    return agent_answer


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
        selected_documents = request_data["selected_documents"]
        agent_answer = _execute_rag(
            query=query, user_id=user_id, selected_documents=selected_documents
        )
        return jsonify(
            {
                "success": True,
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


@app.route("/process_speech_query", methods=["POST"])
def process_speech_query():
    try:
        request_data = json.loads(request.data)
        response = {"success": False}
        user_id = request_data["user_id"]
        selected_documents = request_data["selected_documents"]
        user_query_dict = SpeechQuery.get_latest_query(user_id=user_id)
        if user_query_dict != None and len(user_query_dict) > 0:
            SpeechQuery.set_user_query_state(
                query_id=user_query_dict["query_id"], state="done"
            )
        else:
            return jsonify(response)
        query = user_query_dict["query"] if len(user_query_dict) > 0 else ""
        if query != "":
            agent_answer = _execute_rag(
                query=query, user_id=user_id, selected_documents=selected_documents
            )
            response["success"] = True
            response["answer"] = agent_answer.final_answer
            response["sources"] = agent_answer.function_responses
            response["query"] = query
        return jsonify(response)
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
