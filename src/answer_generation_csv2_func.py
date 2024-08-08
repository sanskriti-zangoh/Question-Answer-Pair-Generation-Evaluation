from depends.model import llm_anton_local_llama3
from depends.vectordb import create_collection_from_documents, get_collection
from depends.document_loader import load_web, load_text, load_pdf, format_docs
from depends.normal_chains import get_chain_1
from depends.prompt import get_ans_prompt_parser_from_question2
from depends.chunking import get_text_splitter
from langchain_core.output_parsers import JsonOutputParser, ListOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv, find_dotenv
from typing import Union, Optional
from langchain_openai import OpenAI
from openai import OpenAI as OpenAIClient
from langchain_core.language_models import BaseLLM
from logging import getLogger, Logger
import pandas as pd

load_dotenv(find_dotenv())
embeddings = OllamaEmbeddings(base_url=os.getenv('EMBEDDING_BASE_URL'), model="all-minilm") 

def answer_generation_csv2_func(logger: Logger, prompt: Optional[PromptTemplate] = None, parser: Optional[Union[JsonOutputParser, StrOutputParser, ListOutputParser]] = None, documents_path: Optional[str] = None, vector_db_collection_name: str = "farming_market", dataframe_path: str = "src/result/test21/qna_overall.csv", llm: Union[OpenAIClient, BaseLLM] = llm_anton_local_llama3, embeddings: OllamaEmbeddings = embeddings, model_name: str = "llama3"):
    if documents_path is not None:
        documents = load_text(documents_path)
        logger.info("STATUS: document loading done")
        text_splitter = get_text_splitter()
        docs = text_splitter.split_documents(documents)
        logger.info("STATUS: document splitting done")
        logger.info("STATUS: total chunks:", len(docs))
        vector_db = create_collection_from_documents(docs, vector_db_collection_name, logger)
        logger.info("STATUS: collection created")
        retriever = vector_db.as_retriever()
        logger.info("STATUS: retriever created")
    else: 
        vector_db = get_collection(vector_db_collection_name)
        logger.info("STATUS: connected to collection")
        retriever = vector_db.as_retriever()
        logger.info("STATUS: retriever created")

    if not prompt or not parser:
        prompt, parser = get_ans_prompt_parser_from_question2()
        logger.warning("STATUS: prompt or parser not provided, using default values")
        logger.info("STATUS: prompt and parser received")

    llm_chain = None
    if isinstance(llm, BaseLLM):
        llm_chain = get_chain_1(llm=llm, prompt=prompt, parser=parser)
        logger.info("STATUS: llm chain created")

    response_dict = {"answer": [], "answer_context": [], "answer_embeddings": [], "answer_context_embeddings": []}
    try:
        data = pd.read_csv(dataframe_path)
    except FileNotFoundError:
        logger.error("ERROR: File not found")
        return

    for index in range(len(data)):
        logger.info(f"STATUS: processing document {index+1} out of {len(data)}")
        relevant_docs = retriever.invoke(input=str(data["question"][index]))
        answer_context_embeddings = embeddings.embed_query(format_docs(relevant_docs))
        while True:
            try:
                if llm_chain:
                    response = llm_chain.invoke({"question": data['question'][index], "rag_context": relevant_docs})
                else:
                    chat_completion = llm.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": f"{prompt.format(question=data['question'][index], rag_context=relevant_docs)}",
                            }
                        ],
                        model=model_name,
                    )

                    response = chat_completion.choices[0].message.content
                    response = parser.parse(response)
                response_dict["answer"].append(response["answer"])
                response_dict["answer_context"].append(format_docs(relevant_docs))
                response_dict["answer_context_embeddings"].append(answer_context_embeddings)
                response_dict["answer_embeddings"].append(embeddings.embed_query(response))
                logger.debug(f"DEBUG: Response:\n{response}")
                break
            except (KeyError, OutputParserException, TypeError) as e:
                logger.error(f"ERROR: {e}. Retrying...")

    data["answer"] = response_dict["answer"]
    data["answer_embeddings"] = response_dict["answer_embeddings"]
    data["answer_context"] = response_dict["answer_context"]
    data["answer_context_embeddings"] = response_dict["answer_context_embeddings"]
    data.to_csv(dataframe_path, index=False)
    logger.info("STATUS: completed and saved")

    return prompt.__dict__
