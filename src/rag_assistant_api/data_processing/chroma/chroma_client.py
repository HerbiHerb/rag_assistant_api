import yaml
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from ..config_schemas import (
    ChromadbConfig,
)

# with open("config/data-config/default.yaml", "r") as file:
#     config_data = yaml.safe_load(file)
# config = ChromadbConfig(**config_data["chroma_db"])
# client = chromadb.Client(
#     Settings(
#         chroma_db_impl="duckdb+parquet", persist_directory=config.persist_directory
#     )
# )
# collection = client.get_collection(config.collection_name)
