from langchain_milvus import Milvus
from pymilvus import Milvus as PyMilvus
from langchain_ollama.embeddings import OllamaEmbeddings
from typing import List, Any, Optional, Dict
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="llama3")

def collection_exists(collection_name: str) -> bool:
    # Initialize Milvus client
    milvus_client = PyMilvus(host="192.168.50.71", port="19532")
    # List all collections
    collections = milvus_client.list_collections()
    return collection_name in collections

def get_collection(collection_name: str) -> Milvus:
    # Initialize Milvus client
    return Milvus(
        collection_name=collection_name,
        embedding_function=embeddings,
        connection_args={"host": "192.168.50.71", "port": "19532"},
    )

def create_collection_from_documents(documents: List[Document], collection_name: str) -> Milvus:
    if collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return get_collection(collection_name)
    else:
        return Milvus.from_documents(
            documents=documents, 
            embedding=embeddings, 
            collection_name=collection_name,
            connection_args={"host": "192.168.50.71", "port": "19532"},
        )

def delete_collection(collection_name: str) -> None:
    milvus_client = PyMilvus()
    milvus_client.drop_collection(collection_name)
    print(f"DELETE: Collection '{collection_name}' deleted.")