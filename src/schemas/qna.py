from pydantic import BaseModel, Field
from typing import List
from langchain_core.documents import Document

class QnA(BaseModel):
    question: str = Field(description="Question")
    answer: str = Field(description="Respective Answer")

class QuestionAnswer(BaseModel):
    question: str 
    answer: str 

class QuestionAnswerContext(BaseModel):
    question: str
    answer: str
    context: str

class QuestionAnswerList(BaseModel):
    question_answer_list: List[QuestionAnswer]

class QuestionAnswerContextList(BaseModel):
    question_answer_context_list: List[QuestionAnswerContext]

class QnAList(BaseModel):
    qna_pairs: List[QnA] = Field(description="List of question and answer pairs")

class QnAListChunk(BaseModel):
    qnas: QnAList
    chunk: Document


class QuestionAnswerRespectiveContext(BaseModel):
    question: str
    answer: str
    question_context: str
    answer_context: str

