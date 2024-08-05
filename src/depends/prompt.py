
from langchain_core.output_parsers import JsonOutputParser, ListOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Tuple

from schemas.question import QuestionList, Question
from schemas.qna import QnAList, QnA, QuestionAnswer, QuestionAnswerList

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

def get_ans_prompt_parser_from_question() -> Tuple[PromptTemplate, StrOutputParser]:
    parser = StrOutputParser()
    prompt = PromptTemplate(
        template="Answer using the information in the given retrieved context from the vector database for the following question: \n{question}. \n\nPlease provide a specific and detailed answer using the information from the retrieved context. The answer should include as much relevant information as possible. If you are unable to answer it, state 'I don't know.' The answer should be informative and contain more than three sentences. Retrieved context from Vector Database:\n{rag_context}",
        input_variables=["question", "rag_context"],
    )
    return prompt, parser

def get_ans_prompt_parser_from_question_simple() -> Tuple[PromptTemplate, StrOutputParser]:
    parser = StrOutputParser()
    prompt = PromptTemplate(
        template="Answer the user's question: \n{question}. \n\nPlease give specific answer instead of the general answer. The answer should be informative and should be more than 3 sentences.",
        input_variables=["question", "rag_context"],
    )
    return prompt, parser