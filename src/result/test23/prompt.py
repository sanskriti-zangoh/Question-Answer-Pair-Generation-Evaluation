{
    "question": {
        "input_variables": ["document_chunk"],
        "partial_variables": {
            "format_instructions": "The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\nthe object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{\"properties\": {\"question\": {\"title\": \"Question\", \"type\": \"string\"}}, \"required\": [\"question\"]}\n```"
        },
        "template": "Please generate a question asking for the key information in the given JSON encoded paragraph:\n{document_chunk}\n\n    Please ask the specific question instead of the general question, like\n    'What is the key information in the given paragraph?'. \n    Also avoid mentioning the 'paragraph' in the question. Instead use the information in the given paragraph.\n    {format_instructions}"
    },
    "answer": {
        "input_variables": ["question", "rag_context"],
        "partial_variables": {
          "format_instructions": "The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\nthe object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{\"properties\": {\"answer\": {\"title\": \"Answer\", \"type\": \"string\"}}, \"required\": [\"answer\"]}\n```"
        },
        "template": "You are given JSON encoded paragraph:\n{rag_context}\n\n\n\n    Answer the question using the information in the given paragraph.\n\n    question: {question}\n\n    Please be specific instead of the generic, like\n    'According to the information provided in the paragraph'.\n    Also avoid mentioning the 'paragraph' in the answer. Instead use the information in the given paragraph.\n    Please generate the answer using as much information as possible.\n    If you are unable to answer the question, the answer will be 'I don't know.'\n    The answer should be informative and should be more than 3 sentences.\n\n    {format_instructions}"
    }         
}