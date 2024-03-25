import os
import sys, os, yaml
import pinecone
import yaml
from .data_processing.config_schemas import PineconeConfig, DataProcessingConfig
from .data_processing.pinecone.pinecone_database_handler import PineconeDatabaseHandler
from .data_processing.pinecone.generate_pinecone_db import generate_database


if __name__ == "__main__":
    os.environ["CREDENTIALS_FP"] = "/Users/marco.schlinger/Code/llmassistantapi/config/credentials/credentials.yaml"
    os.environ["VECTOR_DB_CONFIG_FP"] = "/Users/marco.schlinger/Code/llmassistantapi/config/data-config/default.yaml"
    with open("config/data-config/default.yaml", "r") as file:
        config_data = yaml.safe_load(file)
    data_processing_config = DataProcessingConfig(**config_data["data_processing"])
    pinecone_config = PineconeConfig(**config_data["pinecone_db"])
    database_handler = PineconeDatabaseHandler(
        index=pinecone.Index(pinecone_config.index_name),
        data_processing_config=data_processing_config,
        pinecone_config=PineconeConfig(**config_data["pinecone_db"]),
    )
    generate_database(database_handler=database_handler)
