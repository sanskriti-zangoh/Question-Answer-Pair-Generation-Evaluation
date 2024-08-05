import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from evaluation.store import preprocess_embeddings, parse_array

def cossim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def get_cosine_similarity(dir: str = "result/test5", filename: str = 'qna_context_embeddings.csv', plot_filename: str = 'que_cosine_similarity_plot.png'):

    # load csv
    embedded_queries = pd.read_csv(os.path.join(dir, filename))
    if {'context_embeddings', 'answer_embeddings'}.issubset(embedded_queries.columns):
        embedded_queries['context_embeddings'] = embedded_queries['context_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['answer_embeddings'] = embedded_queries['answer_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['cossim_answer_context'] = embedded_queries.apply(
            lambda row: cossim(row["answer_embeddings"], row["context_embeddings"]), axis=1
        )
    elif {'answer_context_embeddings', 'answer_embeddings'}.issubset(embedded_queries.columns):
        embedded_queries['answer_embeddings'] = embedded_queries['answer_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['answer_context_embeddings'] = embedded_queries['answer_context_embeddings'].apply(lambda x: np.array(json.loads(x)))
        embedded_queries['cossim_answer_context'] = embedded_queries.apply(
            lambda row: cossim(row["answer_embeddings"], row["answer_context_embeddings"]), axis=1
        )
    else:
        print("ERROR: embeddings not found in the csv file")

    # Save the DataFrame to a CSV file
    embedded_queries.to_csv(os.path.join(dir, filename))

    # Plot and save the histogram of cosine similarities
    scores = embedded_queries["cossim_answer_context"].to_list()
    plt.hist(scores, bins=5)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cosine Similarities')
    plt.savefig(os.path.join(dir, plot_filename))

def get_answer_cosine_similarity_csv(dir: str = "result/test5", filename: str = 'qna_overall.csv', plot_filename: str = 'ans_cosine_similarity_plot.png'):
    # Load CSV
    file_path = os.path.join(dir, filename)
    embedded_queries = pd.read_csv(file_path)
    print(f"Loaded file: {file_path}")

    if {'context_embeddings', 'answer_embeddings'}.issubset(embedded_queries.columns):
        try:
            embedded_queries['context_embeddings'] = embedded_queries['context_embeddings'].apply(parse_array)
            embedded_queries['answer_embeddings'] = embedded_queries['answer_embeddings'].apply(parse_array)
        except Exception as e:
            print(f"Error parsing arrays: {e}")
            return

        embedded_queries['cossim_answer_context'] = embedded_queries.apply(
            lambda row: cossim(row["answer_embeddings"], row["context_embeddings"]), axis=1
        )
    elif {'answer_context_embeddings', 'answer_embeddings'}.issubset(embedded_queries.columns):
        try:
            embedded_queries['answer_embeddings'] = embedded_queries['answer_embeddings'].apply(parse_array)
            embedded_queries['answer_context_embeddings'] = embedded_queries['answer_context_embeddings'].apply(parse_array)
        except Exception as e:
            print(f"Error parsing arrays: {e}")
            return

        embedded_queries['cossim_answer_context'] = embedded_queries.apply(
            lambda row: cossim(row["answer_embeddings"], row["answer_context_embeddings"]), axis=1
        )
    else:
        print("ERROR: embeddings not found in the csv file")
        return


    # Plot and save the histogram of cosine similarities
    scores = embedded_queries["cossim_answer_context"].tolist()
    plt.hist(scores, bins=5)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cosine Similarities')
    plt.savefig(os.path.join(dir, plot_filename))
    print(f"Plot saved: {os.path.join(dir, plot_filename)}")

    if {'context_embeddings', 'answer_embeddings'}.issubset(embedded_queries.columns):
        try:
            embedded_queries['context_embeddings'] = embedded_queries['context_embeddings'].apply(lambda x: x.tolist())
            embedded_queries['answer_embeddings'] = embedded_queries['answer_embeddings'].apply(lambda x: x.tolist())
        except Exception as e:
            print(f"Error parsing arrays: {e}")
            return
        
    elif {'answer_context_embeddings', 'answer_embeddings'}.issubset(embedded_queries.columns):
        try:
            embedded_queries['answer_embeddings'] = embedded_queries['answer_embeddings'].apply(lambda x: x.tolist())
            embedded_queries['answer_context_embeddings'] = embedded_queries['answer_context_embeddings'].apply(lambda x: x.tolist())
        except Exception as e:
            print(f"Error parsing arrays: {e}")
            return
    else:
        print("ERROR: embeddings not found in the csv file")
        return
    # Save the DataFrame to a CSV file
    embedded_queries.to_csv(file_path, index=False)
    print(f"Updated file saved: {file_path}")