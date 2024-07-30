from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI

from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

llm_llama3 = OllamaLLM(model="llama3")

llm_gemini = ChatGoogleGenerativeAI(model="gemini-pro", api_key=os.getenv("GOOGLE_API_KEY"))

llm_anton_llama3 = OpenAI(
    api_key=os.getenv("ANTON_MODEL_API_KEY"),
    base_url=os.getenv("ANTON_OLLAMA3_URL"),
    model="/mnt/data/users/khushi/mansion/vllm_deployment/meta-llama-3-8B-awq",
)

llm_anton_llama2 = OpenAI(
    api_key=os.getenv("ANTON_MODEL_API_KEY"),
    base_url=os.getenv("ANTON_OLLAMA2_URL"),
    model="TheBloke/Llama-2-7B-Chat-AWQ",
)