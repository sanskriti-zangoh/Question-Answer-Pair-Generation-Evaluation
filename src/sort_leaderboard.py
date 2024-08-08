import pandas as pd
import os
import argparse

RESULT_PATH = "/Users/sanskrirtisingh/Documents/GitHub/Question-Answer-Pair-Generation-Evaluation/src/result"
SORT_BY_DICT = {
    "quantity": "total_qna_pairs",
    "quality": "overall_avg"
}

def main(sort_by: str = "quantity"):
    try:
        data = pd.read_csv(os.path.join(RESULT_PATH, "leaderboard.csv"))
        data = data.sort_values(by=SORT_BY_DICT[sort_by], ascending=False)
        data.to_csv(os.path.join(RESULT_PATH, "leaderboard.csv"), index=False)
    except FileNotFoundError:
        print("Error: leaderboard.csv not found.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument(
        "--sortby",
        type=str,
        default="quantity",
        choices=["quantity", "quality"],
        help="The sort criteria. Default is 'quantity'."
    )

    args = parser.parse_args()

    main(args.sortby)
