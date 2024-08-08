from calculate_metrics_csv import main as calculate_metrics_csv
from test_on_criteria_csv import test_on_criteria_csv
from overall_evaluation import get_overall_score
import argparse
from typing import Dict
from logging import Logger

def main(logger: Logger, dir: str = "src/result/test20", filename: str = 'qna_overall.csv', llm: str = "anton_local_llama3", model_name: str = "llama3", question_prompt: Dict = {}, answer_prompt: Dict = {}, method: str = "combined", n: int = 3):
    calculate_metrics_csv(dir=dir, filename=filename, logger=logger)
    test_on_criteria_csv(logger=logger, dir=dir, filename=filename)
    get_overall_score(logger=logger, dir=dir, filename=filename, llm=llm, model_name=model_name, question_prompt=question_prompt, answer_prompt=answer_prompt, method=method, n=n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument(
        "--dir",
        type=str,
        default="src/result/24",
        help="The directory where the results are stored. Default is 'src/result/test20'."
    )
    parser.add_argument(
         "--filename",
        type=str,
        default="qna_overall.csv",
        help="The filename of the results. Default is 'qna_overall.csv'."
    )

    args = parser.parse_args()

    main(dir=args.dir, filename=args.filename)