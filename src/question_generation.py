from depends.model import llm_llama3
from depends.document_loader import load_web, load_text, load_pdf, format_docs
from depends.normal_chains import get_chain_1
from depends.prompt import get_que_prompt_parser_from_chunks
from depends.chunking import get_text_splitter
from depends.others import save_to_json_file
from schemas.question import QuestionChunk


documents = load_text("src/data/farming_market.txt")
print("STATUS: document loading done")
text_splitter = get_text_splitter()
docs = text_splitter.split_documents(documents)
print("STATUS: document splitting done")
print("N_CHUNKS:", len(docs))

prompt, parser = get_que_prompt_parser_from_chunks()
print("STATUS: prompt and parser recieved")

llm_chain = get_chain_1(llm=llm_llama3, prompt=prompt, parser=parser)
print("STATUS: llm chain created")

for i, doc in enumerate(docs):
    response = llm_chain.invoke({"document_chunk": format_docs([doc]), "number_of_questions": 5})
    print(f"STATUS: response recieved for chunk {i} of {len(docs)}")
    new_data = QuestionChunk(questions=response, chunk=doc)
    save_to_json_file(response, dir="src/result/test2", filename=f"{i}.json")