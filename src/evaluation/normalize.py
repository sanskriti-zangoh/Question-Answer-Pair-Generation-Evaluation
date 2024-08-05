import pandas as pd
import numpy as np 
from typing import List

def normalize_array(array):
    return array / np.linalg.norm(array)

def normalize_dataframe(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    for column_name in column_names:
        df[column_name + '_normalized'] = df[column_name].apply(lambda x: normalize_array(x))
    return df
