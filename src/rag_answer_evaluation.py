import giskard
import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from dotenv import find_dotenv, load_dotenv
import os

from depends.model import llm_anton_llama2
from evaluation.answer.rag_answer import model_predict_rag_answer, get_giskard_dataset
from depends.normal_chains import get_chain_1
from depends.prompt import get_ans_prompt_parser_from_question
from depends.vectordb import create_collection_from_documents, delete_collection, get_collection
from depends.others import json_to_df
from depends.document_loader import format_docs
from giskard.llm.client.openai import OpenAIClient
from openai import OpenAI

load_dotenv(find_dotenv())
from openai import OpenAI
from giskard.llm.client.openai import OpenAIClient

# Setup the Ollama client with API key and base URL
_client = OpenAI(base_url="http://192.168.50.71:11434/v1/", api_key="ollama")
oc = OpenAIClient(model="llama3", client=_client)
giskard.llm.set_default_client(oc)

vector_db = get_collection("farming_market")
retriever = vector_db.as_retriever()

prompt, parser = get_ans_prompt_parser_from_question()
print("STATUS: prompt and parser received")

llm_chain = get_chain_1(llm=llm_anton_llama2, prompt=prompt, parser=parser)
print("STATUS: llm chain created")


# chose only 5 rows
question_df = json_to_df("/Users/sanskrirtisingh/Documents/GitHub/Question-Answer-Pair-Generation-Evaluation/src/result/test10", "que_context.json")
question_df = question_df.sample(2)

question_df['rag_context'] = question_df['question'].apply(lambda x: format_docs(retriever.invoke(x)))

# Define the prediction function
def prediction_function(inputs):
    df = pd.DataFrame(inputs)
    return model_predict_rag_answer(llm_chain, df, prompt)

# Create the Giskard model with the prediction function
giskard_model = giskard.Model(
    model=prediction_function,
    model_type="text_generation",
    name="Agriculture Question Answering",
    description="This model answers any question about agriculture.",
    feature_names=["question", "rag_context"],
)

giskard_dataset = get_giskard_dataset(question_df)

print(giskard_model.predict(giskard_dataset).prediction)
report = giskard.scan(giskard_model, giskard_dataset)

print(report)