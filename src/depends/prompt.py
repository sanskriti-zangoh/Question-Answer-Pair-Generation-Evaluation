
from langchain_core.output_parsers import JsonOutputParser, ListOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Tuple

from schemas.question import QuestionList, Question
from schemas.qna import QnAList, QnA, QuestionAnswer

def get_que_prompt_parser_from_chunks() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=QuestionList
    )
    prompt = PromptTemplate(
        template="You are given a JSON encoded chunk of a Document:\n{document_chunk}.\n\n\n You are an expert of this document and you are formulating questions from the document to assess the knowledge of a student about content given in this document. \nPlease formulate {number_of_questions} questions to assess knowledge of the document given.\n\n{format_instructions}",
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