
import pandas as pd
import matplotlib.pyplot as plt
import json
import os


def get_question_length_histogram(dir: str = "result/test5", filename: str = 'qna_context.json'):
    # Load the data from the JSON file
    filepath = os.path.join(dir, filename)
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Extract the question lengths
    questions = df["question"].to_list()
    question_len = pd.DataFrame([len(q) for q in questions], columns=["length"])

    # Plot the histogram of question lengths
    question_len.hist(bins=5)
    plt.title("Histogram of Question Lengths")
    plt.xlabel("Question Length")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(dir, "length_histogram.png"))
    plt.show()