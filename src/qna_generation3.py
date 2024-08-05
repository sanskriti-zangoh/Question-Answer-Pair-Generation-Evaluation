from depends.model import llm_anton_local_llama3
from depends.document_loader import load_web, load_text, load_pdf, format_docs
from depends.normal_chains import get_chain_1
from depends.prompt import get_qna_prompt_parser_from_chunks3
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

prompt, parser = get_qna_prompt_parser_from_chunks3()
print("STATUS: prompt and parser recieved")

# llm_chain = get_chain_1(llm=llm_gemini, prompt=prompt, parser=parser)
print("STATUS: llm chain created")

for i, doc in enumerate(docs):
    # try: 
    #     response = llm_chain.invoke({"document_chunk": format_docs([doc]), "number_of_questions": 5})
    #     print(f"STATUS: response received for chunk {i} of {len(docs)}")
    # except OutputParserException: 
    #     print("ERROR: Not proper response format")
    #     continue

    # try:
    #     qna_list = response['question_answer_list']
    # except KeyError:
    #     print("ERROR: Not proper response format")
    #     continue

    # for qna in qna_list:
    #     try:
    #         qna['context'] = doc.page_content
    #     except KeyError:
    #         print("ERROR: Not proper response format")
    #         continue

    #     save_to_json_file(qna, dir="src/result/test9", filename=f"qna_context.json")
    while True:
        print(f"STATUS: processing document {i+1} out of {len(docs)}")
        try:
            # response = llm_chain.invoke({"document_chunk": format_docs([doc]), "number_of_questions": 5})
            chat_completion = llm_anton_local_llama3.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt.format(document_chunk = format_docs([doc]), number_of_questions= 5)}",
                    }
                ],
                model="llama3",
            )

            response = chat_completion.choices[0].message.content
            response = parser.parse(response)

            qna_list=response['question_answer_list']
            # Ensure response is not None and has the expected keys
            for que in qna_list:
                que['context'] = doc.page_content
                if que is not None and 'context' in que.keys():
                    if 'properties' in que.keys():
                        del que['properties']
                    if 'required' in que.keys():
                        del que['required']
                else:
                    raise TypeError("Parsed response is None or missing expected keys.")
                save_to_json_file(que, dir="src/result/test16", filename=f"qna_context.json")
            break

        except (KeyError, OutputParserException, TypeError) as e:
            print(f"ERROR: {e}. Retrying...")