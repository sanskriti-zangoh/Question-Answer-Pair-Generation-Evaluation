import json
from typing import Dict, Optional, Generator, Tuple, List
import os

def save_to_json_file(data: Dict, dir: str = "src/result", filename: str = "test.json") -> None:
    os.makedirs(dir, exist_ok=True)
    file_path = f"{dir}/{filename}"
    
    # Check if file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []
    
    # Assuming the data to be added should be appended to an array in the JSON
    existing_data.append(data)
    
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

def yield_from_ques_json(dir: str) -> Generator[Tuple[str, List[str]], None, None]:
    for filename in os.listdir(dir):
        if filename.endswith('.json'):
            file_path = os.path.join(dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        questions = entry.get("questions", [])
                        for question_data in questions:
                            question = question_data.get("question")
                            topics = question_data.get("topics", [])
                            if question and topics:
                                yield question, topics
                            else:
                                print(f"Skipping empty or malformed entry in file: {file_path}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Error processing file {file_path}: {e}")

def yield_from_data_json(dir: str) -> Generator[Tuple[str, List[str]], None, None]:
    filename = 'combined_data.json'
    file_path = os.path.join(dir, filename)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            data = data['data']
            for entry in data:
                if entry['QueryType']=='Training and Exposure Visits' or entry['QueryType']=='Weather' or entry['QueryType']=='Government Schemes':
                    print(f"Skipping unrelated entry: {file_path}")
                else: 
                    yield entry
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error processing file {file_path}: {e}")

def yield_from_data_evaluation_json(dir: str, filename: str = 'test2.json') -> Generator[Tuple[str, List[str]], None, None]:
    file_path = os.path.join(dir, filename)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                    yield entry
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error processing file {file_path}: {e}")