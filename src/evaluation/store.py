import os
import json
from typing import List
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import pandas as pd

embeddings = OllamaEmbeddings(model="llama3")

def save_embeddings(dir: str = "result/test9", filename: str = 'qna_context.json', csv_filename: str = 'qna_context_embeddings.csv'):
    # Load the data from the JSON file
    filepath = os.path.join(dir, filename)
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)
    
    df['question_embeddings'] = df['question'].apply(lambda x: embeddings.embed_query(x))
    df['answer_embeddings'] = df['answer'].apply(lambda x: embeddings.embed_query(x))
    if 'context' in df.columns:
        df['context_embeddings'] = df['context'].apply(lambda x: embeddings.embed_query(x))
    else:
        df['question_context_embeddings'] = df['question_context'].apply(lambda x: embeddings.embed_query(x))
        df['answer_context_embeddings'] = df['answer_context'].apply(lambda x: embeddings.embed_query(x))

    df.to_csv(os.path.join(dir, csv_filename), index=False)



    
