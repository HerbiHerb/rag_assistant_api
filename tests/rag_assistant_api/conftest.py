import pytest


@pytest.fixture
def config_data():
    config_data = {
        "usage_settings": {
            "agent_type": "OpenAIFunctionsAgent",
            "agent_prompt_config": "rag_assistant",
            "vector_db": "ChromaDatabaseHandler",
            "llm_service": "openai",
            "function_call_prefix": "$FUNCTION_CALL",
        },
        "language_models": {
            "model_name": "gpt-3.5-turbo-0125",
            "embedding_model": "text-embedding-ada-002",
            "temp": 0.0,
        },
        "document_processing": {
            "meta_data_pattern": "(?s)\$META_DATA(.*)\$END_META_DATA",
            "part_seperator": "$PART$",
            "chapter_seperator": "$CHAPTER$",
            "subchapter_seperator": "$SUBCHAPTER$",
            "meta_data_fields": [
                "document_name",
                "autor",
                "date",
                "genre",
                "field",
                "type",
                "user_id",
            ],
        },
        "data_processing": {
            "data_folder_fp": "src/data/raw_data/pdf_files/",
            "batch_size": 10,
            "chunk_size": 512,
            "overlap": 80,
            "embedding_model": "text-embedding-ada-002",
            "embeddings_file_path": "src/data_processing/embeddings.csv",
            "config_credentials": "src/config/credentials",
            "meta_prefix": "meta_",
        },
        "chroma_db": {
            "collection_name": "rag_vector_db",
            "chroma_db_fp": "src/data/chroma_db_instantiation/",
            "dimension": 1536,
            "top_k": 5,
            "metric": "cosine",
        },
    }
    return config_data
