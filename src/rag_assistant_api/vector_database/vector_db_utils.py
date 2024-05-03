import ast
import os
import re
from typing import List
import sys, os, uuid
import yaml
from copy import deepcopy
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from pdfminer.high_level import extract_text
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from ..data_structures.data_structures import (
    PineconeConfig,
    DataProcessingConfig,
    DocumentProcessingConfig,
)
from ..utils.data_processing_utils import (
    split_text_into_parts_and_chapters,
    split_texts_by_keywords,
    split_texts_into_chunks,
    get_embedding,
    check_for_ignore_prefix,
    extract_meta_data,
    remove_meta_data_from_text,
)
from ..base_classes.database_handler import DatabaseHandler
from ..llm_functionalities.embedding_models.embedding_model_factory import (
    create_embedding_model,
)


def generate_text_chunks(
    text: str,
    text_splitter: TokenTextSplitter,
    document_config: DocumentProcessingConfig,
):
    text_chunks_with_chapter = split_text_into_parts_and_chapters(
        text, document_processing_config=document_config
    )
    text_chunks = split_texts_into_chunks(
        text_dicts=text_chunks_with_chapter, text_splitter=text_splitter
    )
    return text_chunks


def extract_text_and_meta_data(
    file_path: str,
    document_config: DocumentProcessingConfig,
):
    with open(file_path, "r", encoding="utf-8") as text_file:
        text = text_file.read()
        meta_data = extract_meta_data(
            extraction_pattern=document_config.meta_data_pattern, document_text=text
        )
        text = remove_meta_data_from_text(text=text)
        return text, meta_data


def process_txt_file(
    file_path: str,
    text_splitter,
    embedding_model: OpenAIEmbeddings,
    database_handler: DatabaseHandler,
    document_config: DocumentProcessingConfig,
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
            extraction_pattern=document_config.meta_data_pattern, document_text=text
        )
        text = remove_meta_data_from_text(text=text)
        text_chunks_with_chapter = split_text_into_parts_and_chapters(
            text, document_processing_config=document_config
        )
        text_chunks = split_texts_into_chunks(
            text_dicts=text_chunks_with_chapter, text_splitter=text_splitter
        )
        upload_chunks_in_batches(
            text_chunks,
            meta_data,
            embedding_model,
            database_handler,
        )


def process_pdf_file(
    file_path: str,
    text_splitter,
    embedding_model: OpenAIEmbeddings,
    database_handler: DatabaseHandler,
    document_config: DocumentProcessingConfig,
):
    """
    Processes a single pdf file by dividing it into text sections,
    creating embeddings for the sections, and uploading the data to a vector database.

    Args:
    - file_path: Path to the file to be processed.
    - meta_file_path: Path to the file with the corresponding meta data.
    - text_splitter: Instance of TokenTextSplitter for text segmentation.
    - embedding_model: Model for generating embeddings.
    - config: Configuration settings for the batch size and other options.
    """
    text = extract_text(file_path)
    text_dict = [{"text": text}]
    text_chunks = split_texts_into_chunks(
        text_dicts=text_dict, text_splitter=text_splitter
    )
    meta_data = {"text": text}
    upload_chunks_in_batches(
        text_chunks,
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
    embedding_model: OpenAIEmbeddings,
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
    database_handler.create_database()
    embedding_model = create_embedding_model(
        llm_service=config_data["usage_settings"]["llm_service"],
        model=config_data["language_models"]["embedding_model"],
    )
    text_splitter = TokenTextSplitter(
        chunk_size=database_handler.data_processing_config.chunk_size,
        chunk_overlap=database_handler.data_processing_config.overlap,
    )
    document_config = DocumentProcessingConfig(**config_data["document_processing"])
    for subdir, dirs, files in os.walk(
        database_handler.data_processing_config.data_folder_fp
    ):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                process_txt_file(
                    file_path=file_path,
                    text_splitter=text_splitter,
                    embedding_model=embedding_model,
                    database_handler=database_handler,
                    document_config=document_config,
                )
            elif file.endswith(".pdf"):
                file_path = os.path.join(subdir, file)
                process_pdf_file(
                    file_path=file_path,
                    text_splitter=text_splitter,
                    embedding_model=embedding_model,
                    database_handler=database_handler,
                    document_config=document_config,
                )


def update_database(
    text: str,
    text_meta_data: dict,
    database_handler: DatabaseHandler,
    document_processing_config: DocumentProcessingConfig,
):
    with open(os.environ["CONFIG_FP"], "r") as file:
        config_data = yaml.safe_load(file)
    embedding_model = OpenAIEmbeddings(
        model=config_data["language_models"]["embedding_model"]
    )
    text_splitter = TokenTextSplitter(
        chunk_size=database_handler.data_processing_config.chunk_size,
        chunk_overlap=database_handler.data_processing_config.overlap,
    )
    text_chunks_with_chapter = split_text_into_parts_and_chapters(
        text=text, document_processing_config=document_processing_config
    )
    text_chunks = split_texts_into_chunks(
        text_dicts=text_chunks_with_chapter, text_splitter=text_splitter
    )
    upload_chunks_in_batches(
        text_chunks,
        text_meta_data,
        embedding_model,
        database_handler,
    )
