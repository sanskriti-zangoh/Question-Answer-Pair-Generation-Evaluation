from pydantic import BaseModel
from typing import List
from langchain_core.documents import Document

class Question(BaseModel):
    question: str

class QuestionList(BaseModel):
    question_list: List[Question]

class QuestionChunk(BaseModel):
    questions: QuestionList
    chunk: str