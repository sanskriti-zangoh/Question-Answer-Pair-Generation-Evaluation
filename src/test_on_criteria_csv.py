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
from langchain_community.embeddings import OllamaEmbeddings
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from logging import Logger

load_dotenv(find_dotenv())

answer_criteria = ['answer_coverage', 'answer_relevancy', 'answer_groundedness']
question_criteria = ['question_fluency']
question_answer_criteria = ['user_relevancy', 'global_relevancy']

# response_dict = {
#     "answer_coverage_score": [], "answer_relevancy_score": [], "answer_groundedness_score": [], "question_fluency_score": [], "user_relevancy_score": [], "global_relevancy_score": [],
#     "answer_coverage_reasoning": [], "answer_relevancy_reasoning": [], "answer_groundedness_reasoning": [], "question_fluency_reasoning": [], "user_relevancy_reasoning": [], "global_relevancy_reasoning": []
# }

def test_on_criteria_csv(logger: Logger, dir = "src/result/test20", filename = 'qna_overall.csv'):
    response_dict = {
        **{criteria + "_score": [] for criteria in answer_criteria},
        **{criteria + "_reasoning": [] for criteria in answer_criteria},
        **{criteria + "_score": [] for criteria in question_criteria},
        **{criteria + "_reasoning": [] for criteria in question_criteria},
        **{criteria + "_score": [] for criteria in question_answer_criteria},
        **{criteria + "_reasoning": [] for criteria in question_answer_criteria},
    }
    data = pd.read_csv(os.path.join(dir, filename))
    for index in range(len(data)):
        for criteria_name in answer_criteria:
            prompt, parser = get_ans_criteria_evaluation_prompt_parser(criteria_name)
            logger.info(f"STATUS: prompt and parser received for {criteria_name}")
            while True:
                try:
                    try:
                        answer_context = data['context'][index]
                    except KeyError:
                        answer_context = data['answer_context'][index]

                    # response = llm_chain.invoke({"question": str(entry["question"]), "answer": str(entry["answer"]), "answer_context": str(answer_context)})
                    chat_completion = llm_anton_local_llama3.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": f"{prompt.format(question=data['question'][index], answer=data['answer'][index], answer_context=answer_context)}",
                            }
                        ],
                        model="llama3",
                    )

                    response = chat_completion.choices[0].message.content
                    response = parser.parse(response)

                    # Ensure response is not None and has the expected keys
                    if response is not None and 'score' in response.keys() and 'reasoning' in response.keys():
                        if 'properties' in response.keys():
                            del response['properties']
                        if 'required' in response.keys():
                            del response['required']
                    else:
                        raise TypeError("Parsed response is None or missing expected keys.")
                    
                    break

                except (KeyError, OutputParserException, TypeError) as e:
                    logger.error(f"ERROR: {e}. Retrying...")

            
            # new_data[criteria_name] = response
            response_dict[criteria_name + "_score"].append(response['score'])
            response_dict[criteria_name + "_reasoning"].append(response['reasoning'])

        for criteria_name in question_criteria:
            prompt, parser = get_que_criteria_evaluation_prompt_parser(criteria_name)
            logger.info(f"STATUS: prompt and parser received for {criteria_name}")
            while True:
                try:
                    try:
                        question_context = data['context'][index]
                    except KeyError:
                        question_context = data['question_context'][index]

                    # response = llm_chain.invoke({"question": str(entry["question"]), "answer": str(entry["answer"]), "answer_context": str(answer_context)})
                    chat_completion = llm_anton_local_llama3.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": f"{prompt.format(question=data['question'][index], question_context=answer_context)}",
                            }
                        ],
                        model="llama3",
                    )

                    response = chat_completion.choices[0].message.content
                    response = parser.parse(response)

                    # Ensure response is not None and has the expected keys
                    if response is not None and 'score' in response.keys() and 'reasoning' in response.keys():
                        if 'properties' in response.keys():
                            del response['properties']
                        if 'required' in response.keys():
                            del response['required']
                    else:
                        raise TypeError("Parsed response is None or missing expected keys.")
                    break

                except (KeyError, OutputParserException, TypeError) as e:
                    logger.error(f"ERROR: {e}. Retrying...")

            
            # new_data[criteria_name] = response
            response_dict[criteria_name + "_score"].append(response['score'])
            response_dict[criteria_name + "_reasoning"].append(response['reasoning'])

        for criteria_name in question_answer_criteria:
            prompt, parser = get_qna_criteria_evaluation_prompt_parser(criteria_name)
            logger.info(f"STATUS: prompt and parser received for {criteria_name}")
            while True:
                try:
                    try:
                        answer_context = data['context'][index]
                        question_context = data['context'][index]
                    except KeyError:
                        answer_context = data['answer_context'][index]
                        question_context = data['question_context'][index]

                    # response = llm_chain.invoke({"question": str(entry["question"]), "answer": str(entry["answer"]), "answer_context": str(answer_context)})
                    chat_completion = llm_anton_local_llama3.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": f"{prompt.format(question=data['question'][index], question_context=question_context, answer=data['answer'][index], answer_context=answer_context)}",
                            }
                        ],
                        model="llama3",
                    )

                    response = chat_completion.choices[0].message.content
                    response = parser.parse(response)

                    # Ensure response is not None and has the expected keys
                    if response is not None and 'score' in response.keys() and 'reasoning' in response.keys():
                        if 'properties' in response.keys():
                            del response['properties']
                        if 'required' in response.keys():
                            del response['required']
                    else:
                        raise TypeError("Parsed response is None or missing expected keys.")
                    break

                except (KeyError, OutputParserException, TypeError) as e:
                    logger.error(f"ERROR: {e}. Retrying...")

            
            # new_data[criteria_name] = response
            response_dict[criteria_name + "_score"].append(response['score'])
            response_dict[criteria_name + "_reasoning"].append(response['reasoning'])

        # save_to_json_file(new_data, dir="src/result/test9", filename=f"overall_criteria_evaluation.json")
        logger.debug("DEBUG:\n{response}")

    for key, value in response_dict.items():
        data[key] = value
    data.to_csv(os.path.join(dir, filename), index=False)
    logger.info("STATUS: criteria evaluation done and saved to {}".format(os.path.join(dir, filename)))



