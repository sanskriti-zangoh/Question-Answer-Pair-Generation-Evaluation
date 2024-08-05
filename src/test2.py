import numpy as np
import pandas as pd

# Read the CSV file
data = pd.read_csv('src/result/test21/qna_overall.csv')
print(list(data["overall"]))
print(np.mean(list(data["overall"])))
print(len(data))

data = pd.read_csv('src/result/test20/qna_overall.csv')
print(list(data["overall"]))
print(np.mean(list(data["overall"])))
print(len(data))

data = pd.read_csv('src/result/test22/qna_overall.csv')
print(list(data["overall"]))
print(np.mean(list(data["overall"])))
print(len(data))
