import ast
import os
import re
from typing import List
import sys, os, uuid
import yaml
from copy import deepcopy
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from ..config_schemas import PineconeConfig, DataProcessingConfig
from ..data_processing_utils import (
    split_txt_file,
    get_embedding,
    check_for_ignore_prefix,
    extract_meta_data,
    remove_meta_data_from_text,
)
from ...base_classes.database_handler import DatabaseHandler


# def remove_meta_data_from_text(text: str):
#     splitted_text = text.split("$END_META_DATA")
#     if splitted_text:
#         return splitted_text[1]


def process_file(
    file_path: str,
    meta_file_path: str,
    text_splitter,
    embedding_model: str,
    database_handler: DatabaseHandler,
):
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
        meta_data = extract_meta_data(
            extraction_pattern=r"(?s)\$META_DATA(.*)\$END_META_DATA", document_text=text
        )
        text = remove_meta_data_from_text(text=text)
        text_chunks_with_chapter = split_txt_file(text, text_splitter=text_splitter)

        upload_chunks_in_batches(
            text_chunks_with_chapter,
            meta_data,
            embedding_model,
            database_handler,
        )


def empty_database(database_handler: DatabaseHandler) -> None:
    """
    Empties the database by deleting and re-creating it

    Args:
        database_handler (DatabaseHandler): Database methods
    """
    try:
        database_handler.delete_database()
    except:
        print("No Database to delete")
    database_handler.create_database()


def upload_chunks_in_batches(
    text_chunks_with_chapters: list[str],
    meta_data: dict,
    embedding_model: str,
    database_handler: DatabaseHandler,
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
    meta_data.update({"text": ""})
    curr_batch = []
    for chunk_dict in text_chunks_with_chapters:
        text = chunk_dict["text"]
        try:
            unique_id = str(uuid.uuid4())
            embedding = get_embedding(text=text, embedding_model=embedding_model)
            meta_data.update(chunk_dict)
            vector_data = {
                "id": unique_id,
                "values": embedding,
                "metadata": deepcopy(meta_data),
            }
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
    with open(os.environ["CONFIG_FP"], "r") as file:
        config_data = yaml.safe_load(file)
    meta_prefix = config_data["data_processing"]["meta_prefix"]
    # set_api_credentials()
    empty_database(database_handler=database_handler)
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    text_splitter = TokenTextSplitter(
        chunk_size=database_handler.data_processing_config.chunk_size,
        chunk_overlap=database_handler.data_processing_config.overlap,
    )

    for subdir, dirs, files in os.walk(
        database_handler.data_processing_config.data_folder_fp
    ):
        for file in files:
            if file.endswith(".txt") and not check_for_ignore_prefix(
                file, ignore_prefix="meta"
            ):
                file_path = os.path.join(subdir, file)
                meta_file_path = os.path.join(subdir, meta_prefix + file)
                process_file(
                    file_path,
                    meta_file_path,
                    text_splitter,
                    embedding_model,
                    database_handler=database_handler,
                )


def update_database(text: str, text_meta_data: dict, database_handler: DatabaseHandler):
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    text_splitter = TokenTextSplitter(
        chunk_size=database_handler.data_processing_config.chunk_size,
        chunk_overlap=database_handler.data_processing_config.overlap,
    )
    text_chunks_with_chapter = split_txt_file(text, text_splitter=text_splitter)
    upload_chunks_in_batches(
        text_chunks_with_chapter,
        text_meta_data,
        embedding_model,
        database_handler,
    )
