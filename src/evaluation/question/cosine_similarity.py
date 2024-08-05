import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from langchain_ollama import OllamaEmbeddings
import numpy as np

def cossim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def get_cosine_similarity(dir: str = "result/test5", filename: str = 'qna_context_embeddings.csv', plot_filename: str = 'que_cosine_similarity_plot.png'):

    # load csv
    embedded_queries = pd.read_csv(os.path.join(dir, filename))
    if {'context_embeddings', 'question_embeddings'}.issubset(embedded_queries.columns):
        embedded_queries['context_embeddings'] = embedded_queries['context_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['question_embeddings'] = embedded_queries['question_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['cossim_question_context'] = embedded_queries.apply(
            lambda row: cossim(row["question_embeddings"], row["context_embeddings"]), axis=1
        )
    elif {'question_context_embeddings', 'question_embeddings'}.issubset(embedded_queries.columns):
        embedded_queries['question_embeddings'] = embedded_queries['question_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['question_context_embeddings'] = embedded_queries['question_context_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['cossim_question_context'] = embedded_queries.apply(
            lambda row: cossim(row["question_embeddings"], row["question_context_embeddings"]), axis=1
        )
    else:
        print("ERROR: embeddings not found in the csv file")

    # Save the DataFrame to a CSV file
    embedded_queries.to_csv(os.path.join(dir, filename))

    # Plot and save the histogram of cosine similarities
    scores = embedded_queries["cossim_question_context"].to_list()
    plt.hist(scores, bins=5)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cosine Similarities')
    plt.savefig(os.path.join(dir, plot_filename))

def get_cosine_similarity_csv(dir: str = "result/test5", filename: str = 'qna_overall.csv', plot_filename: str = 'que_cosine_similarity_plot.png'):

    # load csv
    embedded_queries = pd.read_csv(os.path.join(dir, filename))
    if {'context_embeddings', 'question_embeddings'}.issubset(embedded_queries.columns):
        embedded_queries['context_embeddings'] = embedded_queries['context_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['question_embeddings'] = embedded_queries['question_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['cossim_question_context'] = embedded_queries.apply(
            lambda row: cossim(row["question_embeddings"], row["context_embeddings"]), axis=1
        )
    elif {'question_context_embeddings', 'question_embeddings'}.issubset(embedded_queries.columns):
        embedded_queries['question_embeddings'] = embedded_queries['question_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['question_context_embeddings'] = embedded_queries['question_context_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['cossim_question_context'] = embedded_queries.apply(
            lambda row: cossim(row["question_embeddings"], row["question_context_embeddings"]), axis=1
        )
    else:
        print("ERROR: embeddings not found in the csv file")

    # Save the DataFrame to a CSV file
    embedded_queries.to_csv(os.path.join(dir, filename), index=False)

    # Plot and save the histogram of cosine similarities
    scores = embedded_queries["cossim_question_context"].to_list()
    plt.hist(scores, bins=5)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cosine Similarities')
    plt.savefig(os.path.join(dir, plot_filename))