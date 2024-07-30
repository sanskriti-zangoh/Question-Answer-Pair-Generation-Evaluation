from langchain_community.document_transformers import (
    DoctranQATransformer,
    DoctranTextTranslator,
    Html2TextTransformer,
    MarkdownifyTransformer,
    OpenAIMetadataTagger,
    GoogleTranslateTransformer,
)

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    PDFMinerLoader,
    TextLoader,
    WebBaseLoader,
    JSONLoader
)
from langchain_core.documents import Document
from typing import List

import bs4

def load_text(file_path: str) -> List[Document]:
    return TextLoader(file_path).load()

def load_pdf(file_path: str, extract_images = False) -> List[Document]:
    return PyPDFLoader(file_path, extract_images=extract_images).load()

def load_web(url: str) -> List[Document]:
    return WebBaseLoader(web_path=url).load()

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def load_json(file_path: str, jq_schema: str) -> List[Document]:
    return JSONLoader(file_path, jq_schema=jq_schema, text_content=False).load()