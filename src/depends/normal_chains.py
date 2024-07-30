from depends.model import llm_llama3
from typing import List, Optional, Union, Dict, Any
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import JsonOutputParser, ListOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

default_parser = StrOutputParser()
default_prompt = PromptTemplate(
    template="Question: {question}\nAnswer:",
    input_variables=["question"]
)

def get_chain_1(llm: BaseLLM = llm_llama3, prompt: PromptTemplate = default_prompt, parser: Union[StrOutputParser, JsonOutputParser, ListOutputParser] = default_parser):
    return prompt | llm | parser

