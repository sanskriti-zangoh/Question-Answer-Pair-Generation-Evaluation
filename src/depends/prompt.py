
from langchain_core.output_parsers import JsonOutputParser, ListOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Tuple

from schemas.question import QuestionList, Question
from schemas.qna import QnAList, QnA, QuestionAnswer, QuestionAnswerList
from schemas.answer import Answer

def get_que_prompt_parser_from_chunks() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=QuestionList
    )
    prompt = PromptTemplate(
        template="""Please generate {number_of_questions} questions asking for the key information in the given JSON encoded paragraph:\n{document_chunk}

    Please ask the specific question instead of the general question, like
    'What is the key information in the given paragraph?'. 
    Also avoid mentioning the 'paragraph' in the question. Instead use the information in the given paragraph.
    {format_instructions}""",
        input_variables=["document_chunk", "number_of_questions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

def get_que_prompt_parser_from_chunks_one() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=Question
    )
    prompt = PromptTemplate(
        template="""Please generate a question asking for the key information in the given JSON encoded paragraph:\n{document_chunk}

    Please ask the specific question instead of the general question, like
    'What is the key information in the given paragraph?'. 
    Also avoid mentioning the 'paragraph' in the question. Instead use the information in the given paragraph.
    {format_instructions}""",
        input_variables=["document_chunk", "number_of_questions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

def get_qna_prompt_parser_from_chunks2() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=QuestionAnswerList
    )
    prompt = PromptTemplate(
        template="You are given a JSON encoded chunk of a Document:\n{document_chunk}.\n\n\n You are an expert of this document and you are formulating questions from the document to assess the knowledge of a student about content given in this document. \nPlease formulate {number_of_questions} questions to assess knowledge of the document given. The questions list should contain specific questions instead of the general questions, like 'What is the key information in the given document?'. Also, answer the questions using this document with much information as possible. If you are unable to answer a question, please generate the answer as 'I don't know.' \n\n{format_instructions}",
        input_variables=["document_chunk", "number_of_questions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

def get_qna_prompt_parser_from_chunks() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=QuestionAnswer
    )
    prompt = PromptTemplate(
        template="""Please generate a question asking for the key information in the given JSON encoded paragraph.
    Also answer the questions using the information in the given paragraph.
    Please ask the specific question instead of the general question, like
    'What is the key information in the given paragraph?'.
    Please generate the answer using as much information as possible.
    If you are unable to answer it, please generate the answer as 'I don't know.'
    The answer should be informative and should be more than 3 sentences.
    JSON encoded chunk of a Document:\n{document_chunk}
    {format_instructions}""",
        input_variables=["document_chunk"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

def get_qna_prompt_parser_from_chunks3() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=QuestionAnswerList
    )
    prompt = PromptTemplate(
        template="""Please generate {number_of_questions} questions asking for the key information in the given JSON encoded paragraph:\n{document_chunk}



    Also answer the question list using the information in the given paragraph.
    Please ask the specific question instead of the general question, like
    'What is the key information in the given paragraph?'.
    Also avoid mentioning the 'paragraph' in the question or the answer. Instead use the information in the given paragraph.
    Please generate the respective answers using as much information as possible.
    If you are unable to answer any question, please generate the answer as 'I don't know.'
    The answers should be informative and should be more than 3 sentences.
    {format_instructions}""",
        input_variables=["document_chunk", "number_of_questions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

def get_qna_prompt_parser_from_chunks_one() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=QuestionAnswer
    )
    prompt = PromptTemplate(
        template="""Please generate a question asking for the key information in the given JSON encoded paragraph:\n{document_chunk}



    Also answer the question using the information in the given paragraph.
    Please ask the specific question instead of the general question, like
    'What is the key information in the given paragraph?'.
    Also avoid mentioning the 'paragraph' in the question or the answer. Instead use the information in the given paragraph.
    Please generate the respective answers using as much information as possible.
    If you are unable to answer any question, please generate the answer as 'I don't know.'
    The answers should be informative and should be more than 3 sentences.
    {format_instructions}""",
        input_variables=["document_chunk", "number_of_questions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

def get_ans_prompt_parser_from_question() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=Answer
    )
    prompt = PromptTemplate(
        template="Answer the given question. \n\nquestion: \n{question}. \n\nAvoid mentioning the 'retrieved context' or 'context' or similar terms in the answer. Instead use the information in the given context. The answer should include as much relevant information as possible. If you are unable to answer it, state 'I don't know.' The answer should contain more than three sentences.\n\nRetrieved context from Vector Database:\n{rag_context}\n\n{format_instructions}",
        input_variables=["question", "rag_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

def get_ans_prompt_parser_from_question_simple() -> Tuple[PromptTemplate, StrOutputParser]:
    parser = StrOutputParser()
    prompt = PromptTemplate(
        template="""You are given JSON encoded paragraph:\n{rag_context}



    Answer the question using the information in the given paragraph.

    question: {question}

    Please be specific instead of the generic, like
    'According to the information provided in the paragraph'.
    Also avoid mentioning the 'paragraph' in the answer. Instead use the information in the given paragraph.
    Please generate the answer using as much information as possible.
    If you are unable to answer the question, please generate the answer as 'I don't know.'
    The answer should be informative and should be more than 3 sentences.""",
        input_variables=["question", "rag_context"],
    )
    return prompt, parser

def get_ans_prompt_parser_from_question2() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=Answer
    )
    prompt = PromptTemplate(
        template="""You are given JSON encoded paragraph:\n{rag_context}



    Answer the question using the information in the given paragraph.

    question: {question}

    Please be specific instead of the generic, like
    'According to the information provided in the paragraph'.
    Also avoid mentioning the 'paragraph' in the answer. Instead use the information in the given paragraph.
    Please generate the answer using as much information as possible.
    If you are unable to answer the question, the answer will be 'I don't know.'
    The answer should be informative and should be more than 3 sentences.

    {format_instructions}""",
        input_variables=["question", "rag_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser