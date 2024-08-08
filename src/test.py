from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
from test5 import question, context1, context2

print(f"QUESTION: {question}")

# print("CONTEXTS:")
# print(f"{context1} | {context2}")

print("MODEL: all-minilm")

embeddings = OllamaEmbeddings(model="all-minilm")

sentence1 = embeddings.embed_query(context1)
sentence2 = embeddings.embed_query(context2)

embedded_query = embeddings.embed_query(question)

from sklearn.metrics.pairwise import cosine_similarity
query_sentence1_similarity = cosine_similarity([embedded_query], [sentence1])[0][0]
query_sentence2_similarity = cosine_similarity([embedded_query], [sentence2])[0][0]
print(query_sentence1_similarity, query_sentence2_similarity)

print("MODEL: mxbai-embed-large")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

sentence1 = embeddings.embed_query(context1)
sentence2 = embeddings.embed_query(context2)

embedded_query = embeddings.embed_query(question)

from sklearn.metrics.pairwise import cosine_similarity
query_sentence1_similarity = cosine_similarity([embedded_query], [sentence1])[0][0]
query_sentence2_similarity = cosine_similarity([embedded_query], [sentence2])[0][0]
print(query_sentence1_similarity, query_sentence2_similarity)

print("MODEL: nomic-embed-text")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

sentence1 = embeddings.embed_query(context1)
sentence2 = embeddings.embed_query(context2)

embedded_query = embeddings.embed_query(question)

from sklearn.metrics.pairwise import cosine_similarity
query_sentence1_similarity = cosine_similarity([embedded_query], [sentence1])[0][0]
query_sentence2_similarity = cosine_similarity([embedded_query], [sentence2])[0][0]
print(query_sentence1_similarity, query_sentence2_similarity)

print("MODEL: llama3")

embeddings = OllamaEmbeddings(model="llama3")

sentence1 = embeddings.embed_query(context1)
sentence2 = embeddings.embed_query(context2)

embedded_query = embeddings.embed_query(question)

from sklearn.metrics.pairwise import cosine_similarity
query_sentence1_similarity = cosine_similarity([embedded_query], [sentence1])[0][0]
query_sentence2_similarity = cosine_similarity([embedded_query], [sentence2])[0][0]
print(query_sentence1_similarity, query_sentence2_similarity)


