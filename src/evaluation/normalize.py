import pandas as pd
import numpy as np 
from typing import List

def normalize_llm_criteria_array(array):
    """ Normalize values between 1 and 5 to 0 and 1 """
    return (array - 1) / (5 - 1)

def normalize_array(array):
    """ Normalize values to a range between 0 and 1 """
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val > min_val:
        return (array - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(array)  # or return array if max_val == min_val

def normalize_dataframe(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    """ Normalize specified columns in a DataFrame """
    for column_name in column_names:
        if column_name in ['answer_length', 'question_length']:
            df[column_name + '_normalized'] = normalize_array(df[column_name].values)
        else:
            df[column_name + '_normalized'] = normalize_llm_criteria_array(df[column_name].values)
    return df
