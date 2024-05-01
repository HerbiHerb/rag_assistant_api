import chromadb
import uuid
import os
import openai
import yaml
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from dotenv import load_dotenv
from .vector_database.pinecone.generate_pinecone_db import (
    generate_database,
    update_database,
)
from .data_structures.data_structures import (
    PineconeConfig,
    ChromaDBConfig,
    DataProcessingConfig,
    DocumentProcessingConfig,
)
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from .vector_database.pinecone.pinecone_database_handler import PineconeDatabaseHandler
from .vector_database.chroma_db.chroma_db_database_handler import ChromaDatabaseHandler
from .utils.data_processing_utils import get_embedding


def generate_vector_db():
    load_dotenv()
    with open(
        os.environ["CONFIG_FP"],
        "r",
    ) as file:
        config = yaml.safe_load(file)
    if config["language_models"]["service"] == "OpenAI":
        openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(os.getenv("CONFIG_FP"), "r") as file:
        config_data = yaml.safe_load(file)
    data_processing_config = DataProcessingConfig(**config_data["data_processing"])
    chroma_db_config = ChromaDBConfig(**config_data["chroma_db"])
    database_handler = ChromaDatabaseHandler(
        chroma_db_config=chroma_db_config, data_processing_config=data_processing_config
    )

    test_query = "Der Unterschied ist für Sie so offensichtlich, weil Sie die beiden Paare sehen. Vote und Goat reimen sich, aber sie werden unterschiedlich buchstabiert. Die Teilnehmer hörten nur die Wörter, aber sie wurden auch von der Schreibweise beeinflusst. Sie erkannten deutlich langsamer, dass sich die Wörter reimten, wenn deren Schreibweisen voneinander abwichen. Obgleich die Instruktionen von ihnen nur einen Vergleich der Laute verlangten, verglichen die Teilnehmer auch die Schreibung, und die Nichtübereinstimmung in der irrelevanten Di- nension verlangsamte sie. Die Absicht, eine Frage zu beantworten, rief eine andere Frage hervor, die nicht nur überflüssig war, "
    openai_embeddings = OpenAIEmbeddings(
        model=config_data["language_models"]["embedding_model"]
    )
    test_query_embedding = get_embedding(
        text=test_query, embedding_model=openai_embeddings
    )
    result = database_handler.query(embedding=test_query_embedding, top_k=5)
    # generate_database(database_handler=database_handler)


def read_pdf(pdf_fp: str):
    pass


if __name__ == "__main__":
    read_pdf("src/data/raw_data/pdf_files/a_path_towards_autonomous_ai.pdf")
