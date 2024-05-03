import chromadb
import uuid
import os
import openai
import yaml
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from dotenv import load_dotenv

from rag_assistant_api.base_classes.embedding_base import EmbeddingModel
from ...data_structures.data_structures import (
    ChromaDBConfig,
    DataProcessingConfig,
)
from ...vector_database.chroma_db.chroma_db_database_handler import (
    ChromaDatabaseHandler,
)
from ...llm_functionalities.embedding_models.openai_embeddings import (
    OpenAIEmbeddingModel,
)
import sys, os, uuid
import yaml
from copy import deepcopy
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from pdfminer.high_level import extract_text
from chromadb.config import Settings
from ...utils.data_processing_utils import (
    split_texts_into_chunks,
    get_embedding,
)
from ...base_classes.database_handler import DatabaseHandler
from ...llm_functionalities.embedding_models.embedding_model_factory import (
    create_embedding_model,
)


def insert_text_data(
    text: str,
    text_splitter: TokenTextSplitter,
    embedding_model: EmbeddingModel,
    batch_size: int,
    collection,
):
    text_dict = [{"text": text}]
    text_chunks = split_texts_into_chunks(
        text_dicts=text_dict, text_splitter=text_splitter
    )
    curr_batch = []
    for chunk_dict in text_chunks:
        text = chunk_dict["text"]
        meta_data = {"text": text}
        unique_id = str(uuid.uuid4())
        embedding = get_embedding(text=text, embedding_model=embedding_model)
        vector_data = {
            "id": unique_id,
            "values": embedding,
            "metadata": deepcopy(meta_data),
        }
        curr_batch.append(vector_data)

        if len(curr_batch) == batch_size:
            print("Batch-Hochladen")
            embeddings = [doc_data["values"] for doc_data in curr_batch]
            meta_data = [doc_data["metadata"] for doc_data in curr_batch]
            ids = [str(uuid.uuid4()) for idx in range(len(embeddings))]
            collection.add(embeddings=embeddings, metadatas=meta_data, ids=ids)
            curr_batch = []
    if curr_batch:
        embeddings = [doc_data["values"] for doc_data in curr_batch]
        meta_data = [doc_data["metadata"] for doc_data in curr_batch]
        ids = [str(uuid.uuid4()) for idx in range(len(embeddings))]
        collection.add(embeddings=embeddings, metadatas=meta_data, ids=ids)


def generate_database(
    chroma_db_fp: str,
    collection_name: str,
    data_processing_config: DataProcessingConfig,
):
    """
    Creates and populates a Pinecone database with text data transformed into vectors.

    Args:
    - config: An instance of the PineconeConfig class with the necessary configurations.
    """
    with open(os.environ["CONFIG_FP"], "r") as file:
        config_data = yaml.safe_load(file)
    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_db_fp,
        )
    )
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-ada-002"
        ),
    )
    embedding_model = create_embedding_model(
        llm_service=config_data["usage_settings"]["llm_service"],
        model=config_data["language_models"]["embedding_model"],
    )
    text_splitter = TokenTextSplitter(
        chunk_size=data_processing_config.chunk_size,
        chunk_overlap=data_processing_config.overlap,
    )
    for subdir, dirs, files in os.walk(data_processing_config.data_folder_fp):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(subdir, file)
                text = extract_text(file_path)
                insert_text_data(
                    text=text,
                    text_splitter=text_splitter,
                    embedding_model=embedding_model,
                    batch_size=data_processing_config.batch_size,
                    collection=collection,
                )


def generate_vector_db():
    load_dotenv()
    with open(
        os.environ["CONFIG_FP"],
        "r",
    ) as file:
        config = yaml.safe_load(file)
    if config["usage_settings"]["llm_service"] == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(os.getenv("CONFIG_FP"), "r") as file:
        config_data = yaml.safe_load(file)
    data_processing_config = DataProcessingConfig(**config_data["data_processing"])
    chroma_db_config = ChromaDBConfig(**config_data["chroma_db"])
    database_handler = ChromaDatabaseHandler(
        db_config=chroma_db_config, data_processing_config=data_processing_config
    )
    generate_database(
        chroma_db_fp=chroma_db_config.chroma_db_fp,
        collection_name=chroma_db_config.collection_name,
        data_processing_config=data_processing_config,
    )


def load_vector_db():
    load_dotenv()
    with open(
        os.environ["CONFIG_FP"],
        "r",
    ) as file:
        config = yaml.safe_load(file)
    if config["usage_settings"]["llm_service"] == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
    with open(os.getenv("CONFIG_FP"), "r") as file:
        config_data = yaml.safe_load(file)
    data_processing_config = DataProcessingConfig(**config_data["data_processing"])
    chroma_db_config = ChromaDBConfig(**config_data["chroma_db"])
    database_handler = ChromaDatabaseHandler(
        db_config=chroma_db_config, data_processing_config=data_processing_config
    )
    # test_query = "Wie verwendet Yann Lecun die Theorie über das System 2 in seiner Theorie des World-Models?"
    # test_query = "Appendix: Amortized Inference for Latent Variables Inference in latent variable models consists in performing the optimization zˇ = argminz∈Z Ew(x, y, z). When z is continuous, this may be best performed through"
    test_query = "Amortisierte Inferenz für latente Variablen Die Inferenz in latenten Variablenmodellen besteht in der Durchführung der Optimierung. Wenn z stetig ist, kann dies am besten durch durchgeführt werden"
    openai_embeddings = OpenAIEmbeddingModel(
        embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002")
    )
    # openai_embeddings = OpenAIEmbeddings(
    #     model=config_data["language_models"]["embedding_model"]
    # )
    test_query_embedding = get_embedding(
        text=test_query, embedding_model=openai_embeddings
    )
    result = database_handler.query(embedding=test_query_embedding, top_k=5)
    test = 0


if __name__ == "__main__":
    # generate_vector_db()
    load_vector_db()
