import chromadb, sys, os, uuid, yaml, tqdm, openai
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from ...credentials.setup_credentials import set_api_credentials
from ..config_schemas import ChromadbConfig
from ..data_processing_utils import parse_xml_beautiful_soup, split_txt_file, get_embedding, check_for_ignore_prefix


def make_chromadb_from_xml(config: ChromadbConfig):
    os.environ["CREDENTIALS_FP"] = f"{config.config_credentials}/credentials.yaml"
    set_api_credentials()
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")

    # Initialize chromadb client
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=config.persist_directory))
    collection = client.create_collection(
        config.collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(model_name=config.embedding_model),
    )
    embedding_model = AzureOpenAIEmbeddings(model=config.embedding_model)
    for file in tqdm.tqdm(os.listdir(config.data_folder_fp + config.collection_name)):
        if file.endswith(".xml"):
            # print(file)
            with open(
                os.path.join(config.data_folder_fp + config.collection_name, file),
                "r",
                encoding="utf-8",
            ) as text_file:
                text = text_file.readlines()
                text = "".join(text)
                results = parse_xml_beautiful_soup(text)
                combined_text = results["question"] + "\n\n"
                combined_text += results["answer"]
                try:
                    unique_id = str(uuid.uuid4())
                    collection.add(
                        embeddings=get_embedding(combined_text, embedding_model=embedding_model),
                        metadatas=[{"text": combined_text, "file": file}],
                        ids=[file + "-" + str(unique_id)],
                    )
                except Exception as e:
                    print(e)
                    break


def make_chromadb_from_txt(config: ChromadbConfig):
    os.environ["CREDENTIALS_FP"] = f"{config.config_credentials}/credentials.yaml"
    set_api_credentials()
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")

    # Initialize chromadb client
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=config.persist_directory))
    collection = client.create_collection(
        config.collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(model_name=config.embedding_model),
    )
    embedding_model = AzureOpenAIEmbeddings(model=config.embedding_model)
    text_splitter = TokenTextSplitter(chunk_size=384, chunk_overlap=80)
    visited_files = []

    # for file in tqdm.tqdm(os.listdir(config.data_folder_fp + config.collection_name)):
    for subdir, dirs, files in os.walk(config.data_folder_fp + config.collection_name):
        for file in files:
            if file.endswith(".txt"):
                if check_for_ignore_prefix(file, ignore_prefix="meta") or file in visited_files:
                    continue
                print(file)
                if not file in visited_files:
                    visited_files.append(file)
                with open(
                    os.path.join(subdir, file),
                    "r",
                    encoding="utf-8",
                ) as text_file:
                    text = text_file.readlines()
                    text = " ".join(text)
                    print("start making chunks")
                    text_chunks = split_txt_file(text, text_splitter=text_splitter, chunk_size=384, chunk_overlap=80)
                    for chunk in text_chunks:
                        try:
                            print("adding to collection")
                            unique_id = str(uuid.uuid4())
                            collection.add(
                                embeddings=get_embedding(chunk, embedding_model=embedding_model),
                                metadatas=[{"text": chunk, "file": file}],
                                ids=[file + "-" + str(unique_id)],
                            )
                            print("End")
                        except Exception as e:
                            print(e)
                            break


with open("config/data-config/default.yaml", "r") as file:
    config_data = yaml.safe_load(file)
# make_chromadb_from_xml(ChromadbConfig(**config_data["chroma_db"]), "xml")
make_chromadb_from_txt(ChromadbConfig(**config_data["chroma_db"]))
