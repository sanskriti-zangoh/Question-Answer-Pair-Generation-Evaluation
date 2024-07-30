import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from langchain_ollama import OllamaEmbeddings
import numpy as np

def cossim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def get_cosine_similarity(dir: str = "result/test5", filename: str = 'qna_context.json'):
    # Load the data from the JSON file
    filepath = os.path.join(dir, filename)
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Extract the questions and chunks
    questions = df["question"].to_list()
    chunks = df["context"].to_list()
    
    # Initialize the embedding model
    embedding = OllamaEmbeddings(model="llama3")

    # Embed the questions and chunks
    question_embeddings = np.array([embedding.embed_query(q) for q in questions])
    chunk_embeddings = np.array([embedding.embed_query(c) for c in chunks])

    # Create a DataFrame to hold the embeddings
    embedded_queries = pd.DataFrame({
        "question": questions,
        "context": chunks,
        "question_emb": list(question_embeddings),
        "context_emb": list(chunk_embeddings)
    })

    # Calculate cosine similarities
    embedded_queries["cossim"] = embedded_queries.apply(
        lambda row: cossim(row["question_emb"], row["context_emb"]), axis=1
    )

    # Save the DataFrame to a CSV file
    embedded_queries.to_csv(os.path.join(dir, "cosine_similarity.csv"))

    # Plot and save the histogram of cosine similarities
    scores = embedded_queries["cossim"].to_list()
    plt.hist(scores, bins=5)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cosine Similarities')
    plt.savefig(os.path.join(dir, "cosine_similarity.png"))