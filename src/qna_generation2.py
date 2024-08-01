from depends.model import llm_gemini
from depends.document_loader import load_web, load_text, load_pdf, format_docs
from depends.normal_chains import get_chain_1
from depends.prompt import get_qna_prompt_parser_from_chunks2
from depends.chunking import get_text_splitter
from depends.others import save_to_json_file
from schemas.qna import QnAListChunk, QnAList, QnA, QuestionAnswerContext
from langchain_core.exceptions import OutputParserException

documents = load_text("src/data/farming_market.txt")
print("STATUS: document loading done")
text_splitter = get_text_splitter()
docs = text_splitter.split_documents(documents)
print("STATUS: document splitting done")
print("N_CHUNKS:", len(docs))

prompt, parser = get_qna_prompt_parser_from_chunks2()
print("STATUS: prompt and parser recieved")

llm_chain = get_chain_1(llm=llm_gemini, prompt=prompt, parser=parser)
print("STATUS: llm chain created")

for i, doc in enumerate(docs):
    try: 
        response = llm_chain.invoke({"document_chunk": format_docs([doc]), "number_of_questions": 5})
        print(f"STATUS: response received for chunk {i} of {len(docs)}")
    except OutputParserException: 
        print("ERROR: Not proper response format")
        continue

    try:
        qna_list = response['question_answer_list']
    except KeyError:
        print("ERROR: Not proper response format")
        continue

    for qna in qna_list:
        try:
            qna['context'] = doc.page_content
        except KeyError:
            print("ERROR: Not proper response format")
            continue

        save_to_json_file(qna, dir="src/result/test8", filename=f"qna_context.json")