import pinecone
from pydantic import BaseModel, Field
from ..base_classes.database_handler import DatabaseHandler
from ..data_structures.data_structures import DataProcessingConfig
from ..data_structures.data_structures import (
    DataProcessingConfig,
)
from ..vector_database.pinecone.pinecone_database_handler import PineconeDatabaseHandler
from ..vector_database.chroma_db.chroma_db_database_handler import ChromaDatabaseHandler


class VectorDBFactory:
    factories = {}

    @staticmethod
    def create_vector_db_instance(
        vector_db_cls: str,
        config_data: dict,
    ):
        data_processing_config = DataProcessingConfig(**config_data["data_processing"])
        if not vector_db_cls in VectorDBFactory.factories:
            VectorDBFactory.factories[vector_db_cls] = eval(
                vector_db_cls + ".Factory()"
            )
        return VectorDBFactory.factories[vector_db_cls].create(
            config_data, data_processing_config
        )
