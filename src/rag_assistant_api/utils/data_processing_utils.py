import os
import re
from copy import deepcopy
import yaml
from langchain.text_splitter import TokenTextSplitter
from ..data_structures.data_structures import DocumentProcessingConfig
from ..base_classes.embedding_base import EmbeddingModel


def check_for_ignore_prefix(file_name: str, ignore_prefix: str):
    file_prefix = file_name.split("_")[0]
    if file_prefix == ignore_prefix:
        return True
    return False


def get_embedding(text, embedding_model: EmbeddingModel):
    text = text.replace("\n", " ")
    return embedding_model.generate_embedding(text)


def split_texts_into_parts(texts: list[str], part_seperator: str):
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
):
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
):
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
) -> list[str]:
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
):
    for key in meta_data_dict:
        match_lines = [line for line in extracted_lines if key in line]
        if len(match_lines) > 0:
            extracted_values = re.findall(rf"{key}:?[\s]?(.*)", match_lines[0])
            meta_data_dict[key] = (
                extracted_values[0] if len(extracted_values) > 0 else None
            )
    return meta_data_dict


def extract_meta_data(extraction_pattern: str, document_text: str):
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
    splitted_text = text.split("$END_META_DATA")
    if splitted_text:
        return splitted_text[1]
