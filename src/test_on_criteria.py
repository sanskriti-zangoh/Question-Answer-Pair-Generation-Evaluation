import json
import os
from json import JSONDecodeError

from depends.others import yield_from_ques_json_one, save_to_json_file
from evaluation.schemas.ans_criteria import AnswerCriteria
from depends.model import llm_anton_local_llama3, llm_llama3, llm_anton_llama2
from depends.normal_chains import get_chain_1
from evaluation.prompt import get_ans_evaluation_prompt_parser, get_ans_criteria_evaluation_prompt_parser, get_qna_criteria_evaluation_prompt_parser, get_que_criteria_evaluation_prompt_parser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser, ListOutputParser, StrOutputParser

answer_criteria = ['answer_coverage', 'answer_relevancy', 'answer_groundedness']
question_criteria = ['question_fluency']
question_answer_criteria = ['user_relevancy', 'global_relevancy']



for entry in yield_from_ques_json_one('src/result/test9', 'qna_context.json'):
    new_data = entry
    for criteria_name in answer_criteria:
        prompt, parser = get_ans_criteria_evaluation_prompt_parser(criteria_name)
        print("STATUS: prompt and parser received for", criteria_name)
        while True:
            try:
                try:
                    answer_context = entry['context']
                except KeyError:
                    answer_context = entry['answer_context']

                # response = llm_chain.invoke({"question": str(entry["question"]), "answer": str(entry["answer"]), "answer_context": str(answer_context)})
                chat_completion = llm_anton_local_llama3.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt.format(question=entry['question'], answer=entry['answer'], answer_context=answer_context)}",
                        }
                    ],
                    model="llama3",
                )

                response = chat_completion.choices[0].message.content
                response = parser.parse(response)

                # Ensure response is not None and has the expected keys
                if response is not None and 'score' in response.keys():
                    if 'properties' in response.keys():
                        del response['properties']
                    if 'required' in response.keys():
                        del response['required']
                else:
                    raise TypeError("Parsed response is None or missing expected keys.")
                break

            except (KeyError, OutputParserException, TypeError) as e:
                print(f"ERROR: {e}. Retrying...")

        
        new_data[criteria_name] = response

    for criteria_name in question_criteria:
        prompt, parser = get_que_criteria_evaluation_prompt_parser(criteria_name)
        print("STATUS: prompt and parser received for", criteria_name)
        while True:
            try:
                try:
                    question_context = entry['context']
                except KeyError:
                    question_context = entry['question_context']

                # response = llm_chain.invoke({"question": str(entry["question"]), "answer": str(entry["answer"]), "answer_context": str(answer_context)})
                chat_completion = llm_anton_local_llama3.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt.format(question=entry['question'], question_context=answer_context)}",
                        }
                    ],
                    model="llama3",
                )

                response = chat_completion.choices[0].message.content
                response = parser.parse(response)

                # Ensure response is not None and has the expected keys
                if response is not None and 'score' in response.keys():
                    if 'properties' in response.keys():
                        del response['properties']
                    if 'required' in response.keys():
                        del response['required']
                else:
                    raise TypeError("Parsed response is None or missing expected keys.")
                break

            except (KeyError, OutputParserException, TypeError) as e:
                print(f"ERROR: {e}. Retrying...")

        
        new_data[criteria_name] = response

    for criteria_name in question_answer_criteria:
        prompt, parser = get_qna_criteria_evaluation_prompt_parser(criteria_name)
        print("STATUS: prompt and parser received for", criteria_name)
        while True:
            try:
                try:
                    answer_context = entry['context']
                    question_context = entry['context']
                except KeyError:
                    answer_context = entry['answer_context']
                    question_context = entry['question_context']

                # response = llm_chain.invoke({"question": str(entry["question"]), "answer": str(entry["answer"]), "answer_context": str(answer_context)})
                chat_completion = llm_anton_local_llama3.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt.format(question=entry['question'], question_context=question_context, answer=entry['answer'], answer_context=answer_context)}",
                        }
                    ],
                    model="llama3",
                )

                response = chat_completion.choices[0].message.content
                response = parser.parse(response)

                # Ensure response is not None and has the expected keys
                if response is not None and 'score' in response.keys():
                    if 'properties' in response.keys():
                        del response['properties']
                    if 'required' in response.keys():
                        del response['required']
                else:
                    raise TypeError("Parsed response is None or missing expected keys.")
                break

            except (KeyError, OutputParserException, TypeError) as e:
                print(f"ERROR: {e}. Retrying...")

        
        new_data[criteria_name] = response

    save_to_json_file(new_data, dir="src/result/test9", filename=f"overall_criteria_evaluation.json")
    print("DEBUG:\n", response)


