from typing import Any
import re
from numpy.typing import ArrayLike
from copy import deepcopy
from langchain.text_splitter import TokenTextSplitter
from ..data_structures.data_structures import DocumentProcessingConfig
from ..base_classes.embedding_base import EmbeddingModel


def get_embedding(text: str, embedding_model: EmbeddingModel) -> ArrayLike:
    """
    Function to generate the embedding vector of the input text.

    Args:
        text (str): The chat messages of the current conversation
        encoding_model (EmbeddingModel): The embedding model to generate the embeddings

    Returns:
        ArrayLike: The embedding vector.
    """
    text = text.replace("\n", " ")
    return embedding_model.generate_embedding(text)


def split_texts_into_parts(
    texts: list[str], part_seperator: str
) -> list[dict[str, str]]:
    """
    Function to split input texts into parts.

    Args:
        texts (list[str]): A list of texts to split
        part_seperator (str): A seperating token which marks where the text should be spletted.

    Returns:
        list[dict[str, str]]: The splittings of the texts containing the part name of each text part.
    """
    result_parts = []
    for text in texts:
        parts = text.split(part_seperator)
        for part in parts:
            part_name = ""
            if len(parts) > 1:
                part_name = [
                    section for section in part.split("\n") if len(section) > 0
                ]
                if len(part_name) == 0:
                    continue
                part_name = part_name[0] if len(part_name) > 0 else ""
            result_parts.append({"text": part, "part": part_name})
    return result_parts


def split_texts_by_keywords(
    curr_chunks: list[dict[str, str]], chunk_kw: str, chunk_seperator: str
) -> list[dict[str, str]]:
    """
    Function to split texts chunks and to extract the name of the chunk (text after chunk seperator).

    Args:
        texts (list[str]): A list of texts to split
        chunk_kw (str): The keyword of the chunk.
        chunk_seperator (str): The chunk_seperator to split the chunks into smaller parts.

    Returns:
        list[dict[str, str]]: The splittings of the texts containing the chunks and their chunk names.
    """
    result_parts = []
    for curr_text_dict in curr_chunks:
        text = curr_text_dict["text"]
        chunks = text.split(chunk_seperator)
        for chunk in chunks:
            chunk_name = ""
            if len(chunks) > 1:
                chunk_name = [
                    section for section in chunk.split("\n") if len(section) > 0
                ]
                if len(chunk_name) == 0:
                    continue
                chunk_name = chunk_name[0] if len(chunk_name) > 0 else ""
            text_dict = deepcopy(curr_text_dict)
            text_dict.update({"text": chunk, chunk_kw: chunk_name})
            result_parts.append(text_dict)
    return result_parts


def split_texts_into_chunks(
    text_dicts: list[dict[str, str]], text_splitter: TokenTextSplitter
) -> list[dict[str, str]]:
    """
    Function to split the text dicts into smaller chunks based on the token length.

    Args:
        text_dicts (list[dict[str, str]]): A list of pre-splpitted text chunks (based on seperating tokens)
        text_splitter (TokenTextSplitter): The  langchain text splitter to split the text.

    Returns:
        list[dict[str, str]]: The text chunks splitted into smaller text chunks.
    """
    results = []
    for text_dict in text_dicts:
        text = text_dict["text"]
        text_chunks = text_splitter.split_text(text)
        for text_chunk in text_chunks:
            new_text_dict = deepcopy(text_dict)
            new_text_dict["text"] = text_chunk
            results.append(new_text_dict)
    return results


def split_text_into_parts_and_chapters(
    text: str,
    document_processing_config: DocumentProcessingConfig,
) -> list[dict[str, str]]:
    """
    Function to split the text into parts and chapters.

    Args:
        text (str): The raw input texxt
        document_processing_config (DocumentProcessingConfig): The document processing config object containing all splitting information.

    Returns:
        list[dict[str, str]]: The text chunks splitted into smaller text chunks.
    """
    text = text.replace("\r", "")
    result_chunk_dicts = split_texts_by_keywords(
        curr_chunks=[{"text": text}],
        chunk_kw="part",
        chunk_seperator=document_processing_config.part_seperator,
    )
    result_chunk_dicts = split_texts_by_keywords(
        curr_chunks=result_chunk_dicts,
        chunk_kw="chapter",
        chunk_seperator=document_processing_config.chapter_seperator,
    )
    result_chunk_dicts = split_texts_by_keywords(
        curr_chunks=result_chunk_dicts,
        chunk_kw="subchapter",
        chunk_seperator=document_processing_config.subchapter_seperator,
    )

    return result_chunk_dicts


def extract_meta_data_values(
    extracted_lines: list[str], meta_data_dict: dict[str, str]
) -> dict[str, str]:
    """
    Function to extract the meta data values.

    Args:
        extracted_lines (list[str]): The text lines of the extracted meta data section
        meta_data_dict (dict[str, str]): The dictionary to fill with meta data

    Returns:
        dict[str, str]: The filled dictionary with meta data.
    """
    for key in meta_data_dict:
        match_lines = [line for line in extracted_lines if key in line]
        if len(match_lines) > 0:
            extracted_values = re.findall(rf"{key}:?[\s]?(.*)", match_lines[0])
            meta_data_dict[key] = (
                extracted_values[0] if len(extracted_values) > 0 else None
            )
    return meta_data_dict


def extract_meta_data(extraction_pattern: str, document_text: str):
    """
    Function to extract the meta data.

    Args:
        extraction_pattern (str): The extraction pattern to extract the meta data section.
        document_text (str): The whole document text

    Returns:
        dict[str, str]: The filled meta data dict.
    """
    document_text = document_text.replace("\r", "")
    matches = re.findall(extraction_pattern, document_text)
    meta_data_dict = {
        "document_name": None,
        "autor": None,
        "date": None,
        "genre": None,
        "field": None,
        "type": None,
        "user_id": None,
    }
    if matches:
        extracted_str_lines = matches[0].split("\n")
        extracted_str_lines = [line for line in extracted_str_lines if len(line) > 0]
        meta_data_dict = extract_meta_data_values(
            extracted_lines=extracted_str_lines, meta_data_dict=meta_data_dict
        )
    return meta_data_dict


def remove_meta_data_from_text(text: str):
    """
    Function to remove the meta data section from the input text.

    Args:
        text (str): The whole document text with meta data.

    Returns:
        str: The cleaned document text.
    """
    splitted_text = text.split("$END_META_DATA")
    if splitted_text:
        return splitted_text[1]
