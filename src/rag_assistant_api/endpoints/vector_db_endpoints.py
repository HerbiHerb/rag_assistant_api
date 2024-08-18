import os
import yaml
import json
import openai
from flask import jsonify, request
from ..vector_database.vector_db_factory import VectorDBFactory
from ..init_flask_app import app
from ..local_database.database_models import Document
from ..vector_database.vector_db_utils import (
    generate_database,
    update_database,
)
from ..data_structures.data_structures import (
    DocumentProcessingConfig,
)
from ..utils.data_processing_utils import (
    extract_meta_data,
    remove_meta_data_from_text,
)


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
    request_data = json.loads(request.data)
    user_id = request_data["user_id"]
    uploaded_text = request_data["uploaded_text"]
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
        user_id=user_id,
        document_type=meta_data["type"],
        document_text=uploaded_text,
    )
    return f"Inserted document id {document_id}"
