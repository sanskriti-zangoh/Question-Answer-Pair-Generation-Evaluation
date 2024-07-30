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

class QnAList(BaseModel):
    qna_pairs: List[QnA] = Field(description="List of question and answer pairs")

class QnAListChunk(BaseModel):
    qnas: QnAList
    chunk: Document

