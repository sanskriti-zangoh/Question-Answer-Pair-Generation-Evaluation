import json
import os

from depends.others import yield_from_ques_json_one, save_to_json_file
from evaluation.schemas.ans_criteria import AnswerCriteria
from depends.model import llm_llama3, llm_anton_llama2
from depends.normal_chains import get_chain_1
from evaluation.prompt import get_ans_evaluation_prompt_parser, get_ans_criteria_evaluation_prompt_parser
from langchain_core.exceptions import OutputParserException

prompt, parser = get_ans_criteria_evaluation_prompt_parser('coverage')
print("STATUS: prompt and parser received")

llm_chain = get_chain_1(llm=llm_llama3, prompt=prompt, parser=parser)
print("STATUS: llm chain created")

for entry in yield_from_ques_json_one('src/result/test10', 'qna_context.json'):
    while True:
        try:
            try:
                answer_context = entry['context']
            except KeyError:
                answer_context = entry['answer_context']

            response = llm_chain.invoke({"question": str(entry["question"]), "answer": str(entry["answer"]), "answer_context": str(answer_context)})
            new_data = entry
            new_data["answer_criteria"] = response

            # If the response is valid, break the loop
            break
        except OutputParserException:
            print("ERROR: Not proper response format. Retrying...")

    # save_to_json_file(new_data, dir="src/result/test10", filename=f"ans_criteria.json")
    print("DEBUG:\n", new_data)


