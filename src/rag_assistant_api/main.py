import os
import yaml
import openai
from dotenv import load_dotenv
from .init_flask_app import app
from .data_structures.data_structures import (
    PineconeConfig,
    ChromaDBConfig,
    DataProcessingConfig,
    DocumentProcessingConfig,
    ConfigFileValidator,
)
from .endpoints.assistant_endpoints import *
from .endpoints.sql_database_endboints import *
from .endpoints.vector_db_endpoints import *


def main():
    """
    Main function to initialize the Flask application.
    It sets up environment variables, loads configuration, and starts the Flask app.
    """
    load_dotenv()
    with open(
        os.environ["CONFIG_FP"],
        "r",
    ) as file:
        config_data = yaml.safe_load(file)

    ConfigFileValidator(
        usage_settings=config_data["usage_settings"],
        data_processing_config=DataProcessingConfig(**config_data["data_processing"]),
        document_processing_config=DocumentProcessingConfig(
            **config_data["document_processing"]
        ),
        chroma_db_config=ChromaDBConfig(**config_data["chroma_db"]),
        pinecone_db_config=PineconeConfig(
            api_key=os.getenv("PINECONE_API_KEY"), **config_data["pinecone_db"]
        ),
        prompt_configs_fp=os.getenv("PROMPT_CONFIGS_FP"),
    )
    # Currently only openai and azure is supported
    if config_data["usage_settings"]["llm_service"] in ["openai", "azure"]:
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
