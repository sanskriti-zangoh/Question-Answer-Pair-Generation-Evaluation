import numpy as np
import pandas as pd
import os

from evaluation.weights import weights
from evaluation.normalize import normalize_dataframe
from typing import Dict
from logging import Logger

# Read the CSV file
def get_overall_score(logger: Logger, dir = "src/result/test20", filename = 'qna_overall.csv', llm: str = "anton_local_llama3", model_name: str = "llama3", question_prompt: Dict = {}, answer_prompt: Dict = {}, method: str = "combined", n: int = 3):
    data = pd.read_csv(os.path.join(dir, filename))

    # Ensure metric_fields keys match weights keys after normalization
    metric_fields = [key for key in weights]
    metric_field_already_normalized = ["cossim_question_context", "cossim_answer_context"]

    # Normalize the data
    normalized_df = normalize_dataframe(data, [field for field in metric_fields if field not in metric_field_already_normalized])

    # Generate normalized fields
    normalized_fields = [f'{field}_normalized' for field in metric_fields if field not in metric_field_already_normalized]

    # Include already normalized fields in the normalized fields list
    normalized_fields.extend(metric_field_already_normalized)

    # Ensure that the keys of weights match the normalized fields
    weights_normalized = {key if key in metric_field_already_normalized else f'{key}_normalized': value for key, value in weights.items()}
    logger.info("STATUS: weights are normalized")

    # Calculate the overall score
    data["overall"] = np.average(normalized_df[normalized_fields], axis=1, weights=np.array(list(weights_normalized.values())))

    # Replace NaNs with the mean of the overall scores
    data["overall"].fillna(data["overall"].mean(), inplace=True)

    result_dir = "/".join(dir.split("/")[:-1])
    try:
        leaderboard_data = pd.read_csv(os.path.join(result_dir, 'leaderboard.csv'))
    except FileNotFoundError:
        logger.error("Leaderboard file not found in {}".format(os.path.join(result_dir, 'leaderboard.csv')))
        logger.info("Creating new leaderboard file")
        leaderboard_data = pd.DataFrame(columns=['dir', 'llm', 'model_name', 'method', 'question_prompt', 'answer_prompt', 'number_of_questions_per_context', 'total_qna_pairs', 'question_length_avg', 'answer_length_avg', 'overall_avg'])

    new_row = pd.DataFrame({
        "dir": dir,
        "llm": llm,
        "model_name": model_name,
        "method": method,
        "question_prompt": str(question_prompt),
        "answer_prompt": str(answer_prompt),
        "number_of_questions_per_context": n,
        "total_qna_pairs": len(data),
        "question_length_avg": np.average(data["question_length"]),
        "answer_length_avg": np.average(data["answer_length"]),
        "overall_avg": np.average(data["overall"])
    }, index=[0])
    leaderboard_data = pd.concat([leaderboard_data, new_row], ignore_index=True)

    leaderboard_data.to_csv(os.path.join(result_dir, 'leaderboard.csv'), index=False)
    logger.info("STATUS: leaderboard new row saved to {}".format(os.path.join(result_dir, 'leaderboard.csv')))

    # Save the updated data to CSV
    data.to_csv(os.path.join(dir, filename), index=False)
    logger.info("STATUS: overall score calculation done and saved to {}".format(os.path.join(dir, filename)))
