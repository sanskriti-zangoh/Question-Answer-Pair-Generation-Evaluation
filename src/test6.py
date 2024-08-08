import pandas as pd

# Assuming you already have a DataFrame loaded from a file
try:
    df = pd.read_csv('/Users/sanskrirtisingh/Documents/GitHub/Question-Answer-Pair-Generation-Evaluation/src/result/leaderboard_test.csv')
except FileNotFoundError:
    df = pd.DataFrame(columns=['column1', 'column2'])  # replace with your actual column names

# Create a new row with an explicit index
new_row = pd.DataFrame({
    'column1': [5],  # replace with your actual data
    'column2': ['E']  # replace with your actual data
})

# Concatenate the new row with the existing DataFrame
df = pd.concat([df, new_row], ignore_index=True)

# Save the updated DataFrame back to the file
df.to_csv('/Users/sanskrirtisingh/Documents/GitHub/Question-Answer-Pair-Generation-Evaluation/src/result/leaderboard_test.csv', index=False)
