import math
import numbers
import pandas as pd
import numpy as np
from typing import List, Type, Dict, Tuple
from sqlalchemy import create_engine, select, Column, Integer, BigInteger, Text
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import declarative_base, Session
from abc import ABC, abstractmethod
from openai.lib.azure import AzureOpenAI

Base = declarative_base()


class AbstractTable(Base):
    __abstract__ = "True"

    @property
    @abstractmethod
    def table_name(self):
        """
        property to access the private table name attribute

        :return:
        """
        pass

    @property
    @abstractmethod
    def corresponding_embedded_columns(self) -> Dict:
        """
        Returns a dictionary, where the value of a key is the name of corresponding embedded column

        :return:
        """
        pass

    @property
    def table_columns(self):
        """
        A property to return a list of column names
        :return:
        """
        return self.__table__.columns.keys()


class AbstractPostgresVectorDB(ABC):

    def __init__(self, table: Type[AbstractTable], user: str, password: str, host: str, port: int, db_name: str):
        """

        :param table:
        :param user:
        :param password:
        :param host:
        :param port:
        :param db_name:
        """
        super().__init__()
        self._engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
        self._table: Type[AbstractTable] = table
        Base.metadata.create_all(self._engine, checkfirst=True)

    @abstractmethod
    def get_embeddings(self, text: str or List[str]) -> List:
        """

        :param text:
        :return:
        """
        pass

    def insert_into_table(self, dataframe: pd.DataFrame):
        """

        :param dataframe:
        :return:
        """
        keys = self._table().table_columns
        keys.remove("id")  # TODO: Article dependant!

        with Session(self._engine) as session:
            for index, row in dataframe.iterrows():
                vector_placeholders = len(self._table().corresponding_embedded_columns) * [None]
                values = row.values.tolist() + vector_placeholders
                params = {}
                for k, v in zip(keys, values):
                    if isinstance(v, numbers.Number) and math.isnan(v):
                        params[k] = None
                    else:
                        params[k] = v

                instance = self._table(**params)
                unembedded_vectors: List[Tuple] = []  # [(vector_column_name, corresponding_text_value)]
                for text_col, vector_col in instance.corresponding_embedded_columns.items():
                    text_col_value = getattr(instance, text_col)
                    if text_col_value:  # Only if the value is not None, otherwise Open AI throws exception
                        embedded_vector = self.get_existing_embedded_vector(text_col, text_col_value)
                        if embedded_vector is not None:
                            setattr(instance, vector_col, embedded_vector)
                        else:
                            unembedded_vectors.append((vector_col, text_col_value))

                # If there are text columns to be embedded
                if unembedded_vectors:
                    embedded_vectors = self.get_embeddings([t[1] for t in unembedded_vectors])
                    for col, embedded_vector in zip(unembedded_vectors, embedded_vectors):
                        setattr(instance, col[0], embedded_vector)

                session.add(instance)
                session.commit()

    def get_knn(self, text_column_name, query_text, k=3):
        """

        :param text_column_name: The name of the column which might contain the query text
        :param query_text: The query that is going to be embedded and looked up in the corresponding embedded vector of
        `text_column_name`
        :param k: Number of nearest neighbors to return. Defaults to 3
        :return:
        """
        query_text_vector = self.get_embeddings(query_text)[0]
        embedded_column_name = self._table().corresponding_embedded_columns[text_column_name]
        with Session(self._engine) as session:
            return session.scalars(
                select(self._table)
                .order_by(getattr(self._table, embedded_column_name).cosine_distance(query_text_vector))
                .limit(k)
            ).fetchall()

    def get_existing_embedded_vector(self, column, text) -> np.ndarray or None:
        """
        Given a specific column of type text returns the corresponded embedded vector if existing, otherwise returns
        None

        :param column:
        :param text:
        :return:
        """
        rows = self.__select_where(column, text)
        corresponding_embedded_columns_dict = self._table().corresponding_embedded_columns
        if not rows:
            return None

        return getattr(rows[0], corresponding_embedded_columns_dict[column])

    def __select_where(self, column: str, value: str) -> List[Type[Base]]:
        """
        executes the sql: SELECT * FROM <TABLE> WHERE `column` == `value`

        :param column:
        :param value:
        :return:
        """
        with Session(self._engine) as session:
            stmt = select(self._table).where(getattr(self._table, column) == value)
            return [row[0] for row in session.execute(stmt).fetchall()]


class ArticleTable(AbstractTable):
    # __tablename__ = "hochbau_article"
    __tablename__ = "hochbau_article_new"
    id = Column(Integer, primary_key=True, autoincrement="auto")
    SUPPLIER_PID = Column(Text)
    SUPPLIER_ALT_PID = Column(Text)
    DESCRIPTION_SHORT = Column(Text)
    DESCRIPTION_LONG = Column(Text)
    MANUFACTURER_TYPE_DESCR = Column(Text)
    INTERNATIONAL_PID = Column(BigInteger)
    INTERNATIONAL_PID_TYPE = Column(Text)
    MANUFACTURER_PID = Column(Text)
    DESCRIPTION_CONCATENATED = Column(Text)
    DESCRIPTION_CONCATENATED_VECTOR = Column(Vector(1536))

    @property
    def table_name(self):
        return self.__tablename__

    @property
    def corresponding_embedded_columns(self) -> Dict:
        return {"DESCRIPTION_CONCATENATED": "DESCRIPTION_CONCATENATED_VECTOR"}


class VectorDB(AbstractPostgresVectorDB):
    def __init__(
        self,
        table: Type[AbstractTable],
        user: str,
        password: str,
        host: str,
        port: int,
        db_name: str,
        openai_key: str,
        openai_base: str,
        openai_version: str,
    ):
        super().__init__(table, user=user, password=password, host=host, port=port, db_name=db_name)

        self._client = AzureOpenAI(api_key=openai_key, azure_endpoint=openai_base, api_version=openai_version)

    def batch_insert_into_table(self, dataframe: pd.DataFrame, batch_size: int = 16):
        """
        Embeds and inserts a dataframe, one batch of size `batch_size` at a time

        :param dataframe:
        :param batch_size:
        :return:
        """
        keys = self._table().table_columns
        keys.remove("id")

        with Session(self._engine) as session:
            for i in range(0, len(dataframe), batch_size):
                df_batch = dataframe.iloc[i : i + batch_size, :]
                instances_batch: List = []

                for index, row in df_batch.iterrows():
                    params = {}
                    for key in keys:
                        if key in row:
                            row_value = row[key]
                            params[key] = (
                                row_value
                                if not (isinstance(row_value, numbers.Number) and math.isnan(row_value))
                                else None
                            )
                        else:
                            params[key] = None

                    instances_batch.append(self._table(**params))

                documents: List[str] = df_batch["DESCRIPTION_CONCATENATED"].tolist()
                embedded_documents = self.get_embeddings(documents)

                for instance, embedded_vector in zip(instances_batch, embedded_documents):
                    setattr(instance, "DESCRIPTION_CONCATENATED_VECTOR", embedded_vector)

                session.add_all(instances_batch)
                session.commit()

    def get_embeddings(self, text: str or List[str]) -> List:
        results = self._client.embeddings.create(input=text, model="text-embedding-ada-002")
        return [e.embedding for e in results.data]


# To use the azure postgresql database you can implement it in the following way
# vector_db = VectorDB(
#     table=ArticleTable,
#     user=os.environ["POSTGRES_USER"],
#     password=os.environ["POSTGRES_PASSWORD"],
#     host=os.environ["POSTGRES_HOST"],
#     port=int(os.environ["POSTGRES_PORT"]),
#     db_name=os.environ["POSTGRES_DATABASE_NAME"],
#     openai_key=os.environ["OPENAI_API_KEY"],
#     openai_base=os.environ["AZURE_OPENAI_ENDPOINT"],
#     openai_version=os.environ["OPENAI_API_VERSION"],
# )
# query = """Horizontalisolierung des aufgehenden Mauerwerks herstellen. """
# res = vector_db.get_knn("DESCRIPTION_CONCATENATED", query, k=20)
