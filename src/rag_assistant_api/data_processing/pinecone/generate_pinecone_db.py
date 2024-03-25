import ast
import os
import sys
from typing import List
import sys, os, uuid
import yaml
import json
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from ..config_schemas import PineconeConfig, DataProcessingConfig
from ..data_processing_utils import parse_xml_beautiful_soup, split_txt_file, get_embedding, check_for_ignore_prefix
from ...credentials.setup_credentials import set_api_credentials
from ...base_classes.database_handler import DatabaseHandler


def process_file(file_path: str, meta_file_path: str, text_splitter, embedding_model: str, database_handler: DatabaseHandler):
    """
    Processes a single text file by dividing it into text sections,
    creating embeddings for the sections, and uploading the data to Pinecone.

    Args:
    - file_path: Path to the file to be processed.
    - meta_file_path: Path to the file with the corresponding meta data.
    - text_splitter: Instance of TokenTextSplitter for text segmentation.
    - embedding_model: Model for generating embeddings.
    - config: Configuration settings for the batch size and other options.
    """
    with open(file_path, "r", encoding="utf-8") as text_file:
        text = text_file.read()
        text_chunks = split_txt_file(text, text_splitter=text_splitter)
        try:
            with open(meta_file_path, "r", encoding="utf-8") as meta_file:
                meta_daten = meta_file.read()
        except:
            print('No Meta Data for file:', str(meta_file_path))
            meta_daten = {}
        upload_chunks_in_batches(text_chunks, meta_daten, embedding_model, file_path, meta_file_path, database_handler)


def empty_database(database_handler: DatabaseHandler) -> None:
    """
    Empties the database by deleting and re-creating it
    
    Args:
        database_handler (DatabaseHandler): Database methods
    """
    try:
        database_handler.delete_database()
    except:
        print('No Database to delete')
    database_handler.create_database()


def upload_chunks_in_batches(
    text_chunks: List[str], meta_daten: str, embedding_model: str, file_name: str, meta_file_path: str, database_handler: DatabaseHandler
):
    """
    Uploads text chunks in batches to the Pinecone database.

    Args:
    - text_chunks: List of text chunks for processing.
    - meta_daten: String in json-Format mit Meta-Daten
    - embedding_model: Model for generating embeddings.
    - file_name: Name of the original file from which the text chunks are derived.
    - meta_file_path: Name of the file with the meta data
    - config: Configuration settings for the batch size.
    """
    curr_batch = []
    for chunk in text_chunks:
        try:
            unique_id = str(uuid.uuid4())
            embedding = get_embedding(text=chunk, embedding_model=embedding_model)
            metadata_json = json.loads(meta_daten)
            metadata_json["text"] = chunk
            metadata_json["file"] = file_name
            metadata_json["meta_file"] = meta_file_path
            vector_data = {"id": unique_id, "values": embedding, "metadata": metadata_json}
            curr_batch.append(vector_data)

            if len(curr_batch) == database_handler.data_processing_config.batch_size:
                print("Batch-Hochladen")
                database_handler.upsert(data=curr_batch)
                curr_batch = []
        except Exception as e:
            print(f"Fehler bei der Verarbeitung: {e}")
            break

    if curr_batch:
        database_handler.upsert(data=curr_batch)


def generate_database(database_handler: DatabaseHandler):
    """
    Creates and populates a Pinecone database with text data transformed into vectors.

    Args:
    - config: An instance of the PineconeConfig class with the necessary configurations.
    """
    with open(os.environ["VECTOR_DB_CONFIG_FP"], "r") as file:
        config_data = yaml.safe_load(file)
    meta_prefix = config_data["data_processing"]["meta_prefix"]
    set_api_credentials()
    empty_database(database_handler=database_handler)
    embedding_model = AzureOpenAIEmbeddings(model=database_handler.data_processing_config.embedding_model)
    text_splitter = TokenTextSplitter(
        chunk_size=database_handler.data_processing_config.chunk_size,
        chunk_overlap=database_handler.data_processing_config.overlap,
    )

    for subdir, dirs, files in os.walk(database_handler.data_processing_config.data_folder_fp):
        for file in files:
            if file.endswith(".txt") and not check_for_ignore_prefix(file, ignore_prefix="meta"):
                file_path = os.path.join(subdir, file)
                meta_file_path = os.path.join(subdir, meta_prefix + file)
                process_file(file_path, meta_file_path, text_splitter, embedding_model, database_handler=database_handler)
