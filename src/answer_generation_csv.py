from depends.model import llm_anton_llama2, llm_anton_local_llama3
from depends.vectordb import create_collection_from_documents, delete_collection
from depends.document_loader import load_web, load_text, load_pdf, format_docs
from depends.normal_chains import get_chain_1
from depends.prompt import get_ans_prompt_parser_from_question_simple
from depends.chunking import get_text_splitter
from depends.others import save_to_json_file, yield_from_ques_json, yield_from_ques_json_one
from langchain_core.output_parsers import JsonOutputParser, ListOutputParser, StrOutputParser
from langchain_chroma import Chroma
from langchain_core.exceptions import OutputParserException
from schemas.qna import QuestionAnswerRespectiveContext
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

documents = load_text("src/data/farming_market.txt")
print("STATUS: document loading done")
text_splitter = get_text_splitter()
docs = text_splitter.split_documents(documents)
print("STATUS: document splitting done")
print("N_CHUNKS:", len(docs))

prompt, parser = get_ans_prompt_parser_from_question_simple()
print("STATUS: prompt and parser received")

embeddings = OllamaEmbeddings(base_url=os.getenv('EMBEDDING_BASE_URL'), model="llama3") 
print("STATUS: embeddings initialized")

vector_db = create_collection_from_documents(docs, "farming_market")
retriever = vector_db.as_retriever()

# llm_chain = get_chain_1(llm=llm_anton_llama2, prompt=prompt, parser=parser)
print("STATUS: llm chain created")
response_dict = {"answer": [], "answer_context": [], "answer_embeddings": [], "answer_context_embeddings": []}

import pandas as pd
data = pd.read_csv('src/result/test21/qna_overall.csv')

for index in range(len(data)):
    print(f"STATUS: processing document {index+1} out of {len(data)}")
    entry = data.iloc[index]
    # try:
    #     relevant_docs = retriever.invoke(input=str(entry["question"]))
    #     response = llm_chain.invoke({"question": entry["question"], "rag_context": relevant_docs})

    #     new_data = QuestionAnswerRespectiveContext(
    #         answer=str(response),
    #         question=entry["question"],
    #         question_context=entry["question_context"], 
    #         answer_context=format_docs(relevant_docs)
    #     )

    #     save_to_json_file(new_data.model_dump(), dir="src/result/test10", filename=f"qna_context.json")
    # except OutputParserException: 
    #     print("ERROR: Not proper response format")
    #     continue
    while True:
        try:
            
            relevant_docs = retriever.invoke(input=str(data["question"][index]))
            answer_context_embeddings = embeddings.embed_query(format_docs(relevant_docs))
            # response = llm_chain.invoke({"question": entry["question"], "rag_context": relevant_docs})
            chat_completion = llm_anton_local_llama3.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt.format(question=data['question'][index], rag_context=relevant_docs)}",
                    }
                ],
                model="llama3",
            )

            response = chat_completion.choices[0].message.content
            response = parser.parse(response)
            
            response_dict["answer"].append(response)
            response_dict["answer_context"].append(format_docs(relevant_docs))
            response_dict["answer_context_embeddings"].append(answer_context_embeddings)
            response_dict["answer_embeddings"].append(embeddings.embed_query(response))
            break

        except (KeyError, OutputParserException, TypeError) as e:
            print(f"ERROR: {e}. Retrying...")


data["answer"] = response_dict["answer"]
data["answer_embeddings"] = response_dict["answer_embeddings"]
data["answer_context"] = response_dict["answer_context"]
data["answer_context_embeddings"] = response_dict["answer_context_embeddings"]
data.to_csv('src/result/test21/qna_overall.csv', index=False)