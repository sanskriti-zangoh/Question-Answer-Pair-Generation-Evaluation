from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from typing import List, Any, Optional, Dict
from langchain_core.documents import Document

vectordb_chroma = Chroma(persist_directory="db", embedding_function=OllamaEmbeddings(model="llama3"))

def store_documents_chroma(chroma_db: Chroma, documents: List[Document]) -> None:
    chroma_db.add_documents(documents)

def load_documents_chroma(chroma_db: Chroma, ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Return
        A dict with the keys "ids", "embeddings", "metadatas", "documents".
    """
    return chroma_db.get(ids=ids)

def delete_documents_chroma(chroma_db: Chroma, ids: Optional[List[str]] = None) -> None:
    chroma_db.delete(ids)

def get_vector_db_from_documents(documents: List[Document]) -> Chroma:
    return Chroma.from_documents(documents, embedding=OllamaEmbeddings(model="llama3"))

def vector_db_cleanup(chroma_db: Chroma) -> None:
    chroma_db.delete_collection()