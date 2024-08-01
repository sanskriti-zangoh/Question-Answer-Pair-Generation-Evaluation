from depends.model import llm_anton_llama2
from depends.vectordb import create_collection_from_documents, delete_collection
from depends.document_loader import load_web, load_text, load_pdf, format_docs
from depends.normal_chains import get_chain_1
from depends.prompt import get_ans_prompt_parser_from_question
from depends.chunking import get_text_splitter
from depends.others import save_to_json_file, yield_from_ques_json, yield_from_ques_json_one
from langchain_core.output_parsers import JsonOutputParser, ListOutputParser, StrOutputParser
from langchain_chroma import Chroma
from langchain_core.exceptions import OutputParserException
from schemas.qna import QuestionAnswerRespectiveContext

documents = load_text("src/data/farming_market.txt")
print("STATUS: document loading done")
text_splitter = get_text_splitter()
docs = text_splitter.split_documents(documents)
print("STATUS: document splitting done")
print("N_CHUNKS:", len(docs))

prompt, parser = get_ans_prompt_parser_from_question()
print("STATUS: prompt and parser received")

vector_db = create_collection_from_documents(docs, "farming_market")
retriever = vector_db.as_retriever()

llm_chain = get_chain_1(llm=llm_anton_llama2, prompt=prompt, parser=parser)
print("STATUS: llm chain created")

for entry in yield_from_ques_json_one('src/result/test10'):
    try:
        relevant_docs = retriever.invoke(input=str(entry["question"]))
        response = llm_chain.invoke({"question": entry["question"], "rag_context": relevant_docs})

        new_data = QuestionAnswerRespectiveContext(
            answer=str(response),
            question=entry["question"],
            question_context=entry["question_context"], 
            answer_context=format_docs(relevant_docs)
        )

        save_to_json_file(new_data.model_dump(), dir="src/result/test10", filename=f"qna_context.json")
    except OutputParserException: 
        print("ERROR: Not proper response format")
        continue