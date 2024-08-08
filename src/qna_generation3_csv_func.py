from depends.model import llm_anton_local_llama3
from depends.document_loader import load_web, load_text, load_pdf, format_docs
from depends.normal_chains import get_chain_1
from depends.prompt import get_qna_prompt_parser_from_chunks3
from depends.chunking import get_text_splitter
from depends.others import list_of_dict_to_df_save
from langchain_core.exceptions import OutputParserException
from langchain_community.embeddings import OllamaEmbeddings
import os
from logging import getLogger, Logger
from typing import Optional, Union
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, ListOutputParser, StrOutputParser
from langchain_core.language_models import BaseLLM
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
embeddings = OllamaEmbeddings(base_url=os.getenv('EMBEDDING_BASE_URL'), model="all-minilm") 

def qna_generation3_csv_func(logger: Logger, prompt: Optional[PromptTemplate] = None, parser: Optional[Union[JsonOutputParser, StrOutputParser, ListOutputParser]] = None, documents_path: Optional[str] = None, vector_db_collection_name: str = "farming_market", dataframe_path: str = "src/result/test21/qna_overall.csv", llm: Union[OpenAIClient, BaseLLM] = llm_anton_local_llama3, embeddings: OllamaEmbeddings = embeddings, model_name: str = "llama3", n: int = 3):
    if documents_path is not None:
        documents = load_text(documents_path)
        logger.info("STATUS: document loading done")
        text_splitter = get_text_splitter()
        docs = text_splitter.split_documents(documents)
        logger.info("STATUS: document splitting done")
        logger.info("STATUS: total chunks:", len(docs))
        
    if not prompt or not parser:
        prompt, parser = get_qna_prompt_parser_from_chunks3()
        logger.warning("STATUS: prompt or parser not provided, using default values")
        logger.info("STATUS: prompt and parser received")

    llm_chain = None
    if isinstance(llm, BaseLLM):
        llm_chain = get_chain_1(llm=llm, prompt=prompt, parser=parser)
        logger.info("STATUS: llm chain created")

    response_list = []

    for i, doc in enumerate(docs):
        logger.info(f"STATUS: processing document {i+1} out of {len(docs)}")
        context_embeddings = embeddings.embed_query(doc.page_content)

        while True:
            try:
                if llm_chain is not None:
                    response = llm_chain.invoke({"document_chunk": format_docs([doc]), "number_of_questions": n})
                else:
                    chat_completion = llm.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": f"{prompt.format(document_chunk = format_docs([doc]), number_of_questions= n)}",
                            }
                        ],
                        model=model_name,
                    )

                    response = chat_completion.choices[0].message.content
                    response = parser.parse(response)

                qna_list=response['question_answer_list']
                # Ensure response is not None and has the expected keys
                for qna in qna_list:
                    qna['context'] = doc.page_content
                    if qna is not None and 'context' in qna.keys():
                        if 'properties' in qna.keys():
                            del qna['properties']
                        if 'required' in qna.keys():
                            del qna['required']
                    else:
                        raise TypeError("Parsed response is None or missing expected keys.")
                    qna['context_embeddings'] = context_embeddings
                    qna['question_embeddings'] = embeddings.embed_query(qna['question'])
                    qna['answer_embeddings'] = embeddings.embed_query(qna['answer'])
                    response_list.append(qna)
                break

            except (KeyError, OutputParserException, TypeError) as e:
                logger.error(f"ERROR: {e}. Retrying...")

    dir = "/".join(dataframe_path.split("/")[:-1])
    filename = dataframe_path.split("/")[-1]
    list_of_dict_to_df_save(data=response_list, dir=dir, filename=filename)
    logger.info("STATUS: completed and saved")

    return prompt.__dict__