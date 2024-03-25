from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from bs4 import BeautifulSoup
from langchain.text_splitter import TokenTextSplitter


def check_for_ignore_prefix(file_name: str, ignore_prefix: str):
    file_prefix = file_name.split("_")[0]
    if file_prefix == ignore_prefix:
        return True
    return False


def get_embedding(text, embedding_model: AzureOpenAIEmbeddings):
    text = text.replace("\n", " ")
    return embedding_model.embed_query(text)


def parse_xml_beautiful_soup(content):
    bs_content = BeautifulSoup(content, "lxml")
    title_node = bs_content.find("title")
    answer_node = bs_content.find("answer")
    keyword_node = bs_content.find("keywords")

    title_text = extract_text(title_node)
    answer_text = extract_text(answer_node)
    keyword_text = extract_text(keyword_node)
    result_dict = {
        "question": title_text,
        "answer": answer_text,
        "keywords": keyword_text,
    }
    return result_dict


def split_txt_file(text: str, text_splitter: TokenTextSplitter, chunk_size: int = 512, chunk_overlap=50) -> List[str]:
    return text_splitter.split_text(text)


def extract_text(element):
    text = ""
    for child in element.children:  # Durchlaufe alle direkten Kinder des Elements
        if child.name is not None:  # Überprüfe, ob das Kind ein Element ist (nicht nur Text)
            text += extract_text(child) + " "
        elif child != None and len(child) > 0:
            text += child + " "
    return text
