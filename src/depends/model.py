from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from openai import OpenAI as OpenAIClient

from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

llm_llama3 = OllamaLLM(model="llama3")

llm_gemini = ChatGoogleGenerativeAI(model="gemini-pro", api_key=os.getenv("GOOGLE_API_KEY"))

llm_anton_llama3 = OpenAI(
    api_key=os.getenv("ANTON_MODEL_API_KEY"),
    base_url=os.getenv("ANTON_OLLAMA3_URL"),
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
)

llm_anton_llama2 = OpenAI(
    api_key=os.getenv("ANTON_MODEL_API_KEY"),
    base_url=os.getenv("ANTON_OLLAMA2_URL"),
    model="TheBloke/Llama-2-7B-Chat-AWQ",
)

llm_anton_local_llama3 = OpenAIClient(
    api_key="ollama",
    base_url=os.getenv("ANTON_LOCAL_OLLAMA3_URL"),
)

llm_anton_llama3_1 = OpenAIClient(
    api_key=os.getenv("ANTON_MODEL_API_KEY"),
    base_url=os.getenv("ANTON_OLLAMA3_URL"),
)

# llm_anton_local_llama3 = OpenAI(
#     api_key="ollama",
#     base_url=os.getenv("ANTON_LOCAL_OLLAMA3_URL"),
#     model="llama3",
# )