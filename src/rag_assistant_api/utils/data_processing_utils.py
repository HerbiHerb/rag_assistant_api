from typing import List
import re
from langchain_openai import OpenAIEmbeddings
from bs4 import BeautifulSoup
from langchain.text_splitter import TokenTextSplitter


def check_for_ignore_prefix(file_name: str, ignore_prefix: str):
    file_prefix = file_name.split("_")[0]
    if file_prefix == ignore_prefix:
        return True
    return False


def get_embedding(text, embedding_model: OpenAIEmbeddings):
    text = text.replace("\n", " ")
    return embedding_model.embed_query(text)


def split_txt_file(text: str, text_splitter: TokenTextSplitter) -> list[str]:
    result_chunks = []
    chapter_chunks = text.split("$CHAPTER$")
    for chapter in chapter_chunks:
        chapter_name = [section for section in chapter.split("\n") if len(section) > 0]
        if len(chapter_name) == 0:
            continue
        chapter_name = chapter_name[0]
        text_chunks = text_splitter.split_text(chapter)
        for chunk in text_chunks:
            result_chunks.append({"text": chunk, "chapter": chapter_name})

    return result_chunks


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


def split_text_on_chapters():
    pass


def remove_meta_data_from_text(text: str):
    splitted_text = text.split("$END_META_DATA")
    if splitted_text:
        return splitted_text[1]
