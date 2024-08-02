import json
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Define connection parameters
milvus_host = os.getenv('MILVUS_HOST')
milvus_port = os.getenv('MILVUS_PORT')
# Connect to Milvus server
try:
    connections.connect(alias="default", host=milvus_host, port=milvus_port)
    logger.info("Connected to Milvus server.")
except Exception as e:
    logger.error(f"Failed to connect to Milvus server: {e}")
    exit(1)
# List collections in Milvus
try:
    collections = utility.list_collections()
    logger.info(f"Collections in Milvus: {collections}")
except Exception as e:
    logger.error(f"Failed to list collections: {e}")
    exit(1)
# Load JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        raw_content = file.read()
        logger.info(f"Raw file content (first 500 chars): {raw_content[:500]}")
        data = json.loads(raw_content)
    logger.info(f"Loaded JSON data. Number of messages: {len(data.get('messages', []))}")
    return data["messages"]
# Process data
def process_data(messages):
    doc_count = 0
    for item in messages:
        title = item.get("metadata", {}).get("title", "No Title")
        content = item.get("page_content", "No Content")
        doc_count += 1
        yield Document(page_content=content, metadata={"title": title})
    logger.info(f"Processed {doc_count} documents")
# Load and process data
data_file = "/Users/japjeetsinghchhabra/Desktop/Test ofr speed/vikaspedia_agriculture_using_unstructured.json"
messages = load_json(data_file)
documents = list(process_data(messages))  # Convert to list to get accurate count
logger.info(f"Total number of documents: {len(documents)}")
# Initialize Ollama embeddings with Anton server URL
embeddings = OllamaEmbeddings(base_url=os.getenv('EMBEDDING_BASE_URL'), model="llama3")
# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
def batch_split_documents(documents, batch_size=1000):
    batch = []
    for doc in documents:
        batch.append(doc)
        if len(batch) == batch_size:
            splits = text_splitter.split_documents(batch)
            yield from splits
            batch = []
    if batch:
        splits = text_splitter.split_documents(batch)
        yield from splits
# Use a generator for splitting documents
all_splits = batch_split_documents(documents)
# Function to check embedding dimension
def check_embedding_dimension(docs, expected_dim=4096):
    for doc in list(docs)[:5]:  # Check first 5 documents
        embedding = embeddings.embed_query(doc.page_content)
        if len(embedding) != expected_dim:
            logger.warning(f"Embedding dimension mismatch. Expected {expected_dim}, got {len(embedding)}")
            return False
    return True
# Check embedding dimension before inserting
if not check_embedding_dimension(all_splits):
    logger.error("Embedding dimension mismatch. Please check your embedding model.")
    exit(1)
# Create collection
collection_name = "document"
dim = 4096
def create_collection(name, dim):
    if name in utility.list_collections():
        utility.drop_collection(name)
        logger.info(f"Dropped existing collection: {name}")
    fields = [
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, "Document collection")
    collection = Collection(name, schema)
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index("vector", index_params)
    logger.info(f"Created collection: {name}")
    return collection
create_collection(collection_name, dim)
# Convert generator to list to ensure it's not exhausted
all_splits = list(batch_split_documents(documents))
logger.info(f"Total number of splits: {len(all_splits)}")
# Add sample logging
if all_splits:
    logger.info(f"Sample split - Title: {all_splits[0].metadata.get('title')}")
    logger.info(f"Sample split - Content (first 100 chars): {all_splits[0].page_content[:100]}")
else:
    logger.warning("No splits generated")
def insert_batches(collection_name, all_splits, batch_size=1000):
    vectorstore = None
    batch = []
    total_docs = 0
    for i, doc in enumerate(all_splits, 1):
        batch.append(doc)
        if len(batch) == batch_size:
            try:
                if vectorstore is None:
                    logger.info(f"Creating new Milvus vectorstore with batch of {len(batch)} documents")
                    vectorstore = Milvus.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        collection_name=collection_name,
                        connection_args={"host": milvus_host, "port": milvus_port}
                    )
                else:
                    logger.info(f"Adding batch of {len(batch)} documents to existing vectorstore")
                    vectorstore.add_documents(batch)
                total_docs += len(batch)
                logger.info(f"Inserted batch {i // batch_size}. Total documents inserted: {total_docs}")
                batch = []
            except Exception as e:
                logger.error(f"Failed to insert batch ending at document {i}")
                logger.error(f"Error: {e}")
                logger.info("Attempting to insert documents one by one")
                for single_doc in batch:
                    try:
                        if vectorstore is None:
                            vectorstore = Milvus.from_documents(
                                documents=[single_doc],
                                embedding=embeddings,
                                collection_name=collection_name,
                                connection_args={"host": milvus_host, "port": milvus_port}
                            )
                        else:
                            vectorstore.add_documents([single_doc])
                        total_docs += 1
                        logger.info(f"Successfully inserted document {total_docs}")
                    except Exception as doc_e:
                        logger.error(f"Failed to insert document: {doc_e}")
                        logger.error(f"Problematic document - Title: {single_doc.metadata.get('title')}")
                        logger.error(f"Content length: {len(single_doc.page_content)}")
            batch = []
    if batch:
        try:
            if vectorstore is None:
                logger.info(f"Creating new Milvus vectorstore with final batch of {len(batch)} documents")
                vectorstore = Milvus.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=collection_name,
                    connection_args={"host": milvus_host, "port": milvus_port}
                )
            else:
                logger.info(f"Adding final batch of {len(batch)} documents to existing vectorstore")
                vectorstore.add_documents(batch)
            total_docs += len(batch)
            logger.info(f"Inserted final batch. Total documents inserted: {total_docs}")
        except Exception as e:
            logger.error("Failed to insert final batch")
            logger.error(f"Error: {e}")
    return total_docs
# Check collection status and entity count
try:
    collection = Collection(collection_name)
    collection.load()
    entity_count = collection.num_entities
    logger.info(f"Collection '{collection_name}' loaded. Entity count: {entity_count}")
except Exception as e:
    logger.error(f"Failed to check collection status: {e}")
# Main execution
try:
    total_inserted = insert_batches(collection_name, all_splits)
    logger.info(f"All documents have been successfully stored in Milvus. Total documents inserted: {total_inserted}")
except Exception as e:
    logger.error(f"An error occurred during document insertion: {e}")