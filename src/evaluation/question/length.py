
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from logging import Logger
from typing import Optional


def get_question_length_histogram(dir: str = "result/test5", filename: str = 'qna_context_embeddings.csv', plot_filename: str = 'que_length_histogram.png'):

    # Load to pandas Datafram
    df = pd.read_csv(os.path.join(dir, filename))

    # Extract the question lengths
    questions = df["question"].to_list()
    df["question_length"] = df["question"].apply(lambda x: len(x))
    df.to_csv(os.path.join(dir, filename), index=False)
    question_len = pd.DataFrame([len(q) for q in questions], columns=["length"])

    # Plot the histogram of question lengths
    question_len.hist(bins=5)
    plt.title("Histogram of Question Lengths")
    plt.xlabel("Question Length")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(dir, plot_filename))
    plt.show()

def get_question_length_histogram_csv(dir: str = "result/test5", filename: str = 'qna_overall.csv', plot_filename: str = 'que_length_histogram.png', logger: Optional[Logger] = None):

    # Load to pandas Datafram
    df = pd.read_csv(os.path.join(dir, filename))

    # Extract the question lengths
    questions = df["question"].to_list()
    df["question_length"] = df["question"].apply(lambda x: len(x))
    df.to_csv(os.path.join(dir, filename), index=False)
    question_len = pd.DataFrame([len(q) for q in questions], columns=["length"])

    # Plot the histogram of question lengths
    question_len.hist(bins=5)
    plt.title("Histogram of Question Lengths")
    plt.xlabel("Question Length")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(dir, plot_filename))
    plt.close()
    if logger:
        logger.info("Histogram of Question Lengths saved to {}".format(os.path.join(dir, plot_filename)))