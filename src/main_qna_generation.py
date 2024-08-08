import argparse
from depends.model import llm_anton_local_llama3, llm_anton_llama2, llm_anton_llama3, llm_gemini, llm_llama3
from depends.document_loader import load_web, load_text, load_pdf, format_docs, load_json
from answer_generation_csv2_func import answer_generation_csv2_func
from generic_question_generation_csv_func import generate_question_generation_csv_func
from generic_qna_generation_csv_func import generate_qna_generation_csv_func
from qna_generation3_csv_func import qna_generation3_csv_func
from question_generation_csv_func import question_generation_csv_func
from logging import getLogger, Logger
import os
from main_qna_evaluation import main as main_qna_evaluation

DATA_PATH = "/Users/sanskrirtisingh/Documents/GitHub/Question-Answer-Pair-Generation-Evaluation/src/data"
RESULT_PATH = "/Users/sanskrirtisingh/Documents/GitHub/Question-Answer-Pair-Generation-Evaluation/src/result"

LLM_DICT = {
    "llm_anton_llama2": llm_anton_llama2,
    "llm_anton_llama3.1": llm_anton_llama3,
    "llm_anton_local_llama3": llm_anton_local_llama3,
    "llm_gemini": llm_gemini,
    "llm_llama3": llm_llama3
}

MODEL_DICT = {
    "llm_anton_llama2": "TheBloke/Llama-2-7B-Chat-AWQ",
    "llm_anton_llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llm_anton_local_llama3": "llama3",
    "llm_gemini": "gemini-pro",
    "llm_llama3": "llama3"
}

def main(method, generic, llm, collection_name, filename, n, model_name):
    model_name = MODEL_DICT[llm]
    logger = getLogger("qna_generation")
    logger.info(f"Method: {method}")
    logger.info(f"Generic: {generic}")
    logger.info(f"LLM: {llm}")
    logger.info(f"Collection Name: {collection_name}")
    logger.info(f"Filename: {filename}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"N: {n}")
    result_data_path, result_data_filename = f"{RESULT_PATH}/{len(os.listdir(RESULT_PATH)) + 1}", "qna_overall.csv"
    result_data_file_path = f"{result_data_path}/{result_data_filename}"
    logger.info(f"Result Data Path: {result_data_path}")
    logger.info(f"Result Data Filename: {result_data_filename}")

    args_dict_with_n = {
        "logger": logger,
        "documents_path": f"{DATA_PATH}/{filename}",
        "vector_db_collection_name": collection_name,
        "dataframe_path": result_data_file_path,
        "llm": LLM_DICT[llm],
        "n": n
    }
    args_dict_without_n = {
        "logger": logger,
        "documents_path": f"{DATA_PATH}/{filename}",
        "vector_db_collection_name": collection_name,
        "dataframe_path": result_data_file_path,
        "llm": LLM_DICT[llm]
    }
    question_prompt, answer_prompt = {}, {}
    if method == "separate":
        if generic:
            question_prompt = generate_question_generation_csv_func(**args_dict_without_n)
        else:
            question_prompt = question_generation_csv_func(**args_dict_with_n)
        answer_prompt = answer_generation_csv2_func(**args_dict_without_n)
    else:
        if generic:
            question_prompt = generate_qna_generation_csv_func(**args_dict_without_n)
        else:
            question_prompt = qna_generation3_csv_func(**args_dict_with_n)

        answer_prompt = question_prompt

    method_name = method
    if generic:
        method_name += "_generic"
        n=1

    main_qna_evaluation(logger=logger, dir=result_data_path, filename=result_data_filename, llm=llm, model_name=model_name, question_prompt=question_prompt, answer_prompt=answer_prompt, method=method_name, n=n)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument(
        "--method",
        type=str,
        choices=["separate", "combined"],
        default="combined",
        help="Method to generate Q&A pairs, either 'separate' or 'combined'. Default is 'combined'."
    )
    parser.add_argument(
        "--generic",
        action='store_true',
        help="Flag to indicate if the Q&A generation should be generic."
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="farming_market",
        help="Name of the collection in the case of 'separate' method. Default is 'farming_market'."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="farming_market.txt",
        help="Name of the file to be used as the raw data. Default is 'farming_market.txt'."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of questions per chunk. Default is 3."
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["llm_anton_llama2", "llm_anton_llama3.1", "llm_anton_local_llama3", "llm_gemini", "llm_llama3"],
        default="anton_local_llama3",
        help="The llm model to generate the QnA pairs. Default is 'anton_local_llama3'."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["llama3", "gemini-pro", "TheBloke/Llama-2-7B-Chat-AWQ", "meta-llama/Meta-Llama-3.1-8B-Instruct"],
        default="llama3",
        help="The llm model to generate the QnA pairs. Default is 'anton_local_llama3'."
    )

    args = parser.parse_args()
    
    main(args.method, args.generic, args.llm, args.collection_name, args.filename, args.n, args.model_name)

