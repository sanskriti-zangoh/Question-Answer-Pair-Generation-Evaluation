from depends.model import llm_anton_local_llama3
from depends.document_loader import load_web, load_text, load_pdf, format_docs
from depends.normal_chains import get_chain_1
from depends.prompt import get_qna_prompt_parser_from_chunks_one
from depends.chunking import get_text_splitter
from depends.others import list_of_dict_to_df_save
from schemas.qna import QnAListChunk, QnAList, QnA, QuestionAnswerContext
from langchain_core.exceptions import OutputParserException
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

prompt, parser = get_qna_prompt_parser_from_chunks_one()
print("STATUS: prompt and parser received")

embeddings = OllamaEmbeddings(base_url=os.getenv('EMBEDDING_BASE_URL'), model="llama3") 
print("STATUS: embeddings initialized")

# llm_chain = get_chain_1(llm=llm_gemini, prompt=prompt, parser=parser)
print("STATUS: llm chain created")
response_list = []

for i, doc in enumerate(docs):
    print(f"STATUS: processing document {i+1} out of {len(docs)}")
    context_embeddings = embeddings.embed_query(doc.page_content)
    while True:
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

            qna=response
            # Ensure response is not None and has the expected keys
            qna['context'] = doc.page_content
            if qna is not None and 'answer' in qna.keys() and 'question' in qna.keys():
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
            print(f"ERROR: {e}. Retrying...")

list_of_dict_to_df_save(data=response_list, dir="src/result/test22", filename="qna_overall.csv")