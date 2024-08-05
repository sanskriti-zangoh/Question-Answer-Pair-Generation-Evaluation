import numpy as np
import pandas as pd

from evaluation.weights import weights
from evaluation.normalize import normalize_dataframe

# Read the CSV file
data = pd.read_csv('src/result/test21/qna_overall.csv')

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

# Calculate the overall score
data["overall"] = np.average(normalized_df[normalized_fields], axis=1, weights=np.array(list(weights_normalized.values())))

# replace nan by mean
data["overall"] = data["overall"].fillna(data["overall"].mean())

# Save the updated data to CSV
data.to_csv('src/result/test21/qna_overall.csv', index=False)