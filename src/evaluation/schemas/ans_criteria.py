from pydantic import BaseModel, Field


class AnswerCoverage(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


class AnswerRelevancy(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


class AnswerGroundedness(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str

class AnswerCriteria(BaseModel):
    coverage: AnswerCoverage = Field(description="Coverage score is to gauge whether the generated answers can be directly extracted from the provided context and how well the model avoids hallucinating. Rate the answerability of each Q&A pair on a scale of 1 to 5. A higher score indicates that the answer can be more reliably extracted from the context, ensuring the model’s output is grounded in the available information.")
    relevancy: AnswerRelevancy = Field(description="Relevance measures how well the answer addresses the main aspects of the question based on the context. The metric rates from 1 to 5, where 5 means the answer has perfect relevance.")
    groundedness: AnswerGroundedness = Field(description="Groundedness is the metric that defines weather the answer follows logically from the information contained in the context or not and provides and integer score (1-5) to determine how grounded the answer is.")

answer_criteria = {
    "answer_coverage": {
        "description": "Coverage score is to gauge whether the generated answers can be directly extracted from the provided context and how well the model avoids hallucinating. Rate the answerability of each Q&A pair on a scale of 1 to 5. A higher score indicates that the answer can be more reliably extracted from the context, ensuring the model’s output is grounded in the available information.",
        "json_schema": AnswerCoverage
    },
    "answer_relevancy": {
        "description": "Relevance measures how well the answer addresses the main aspects of the question based on the context. The metric rates from 1 to 5, where 5 means the answer has perfect relevance.",
        "json_schema": AnswerRelevancy
    },
    "answer_groundedness": {
        "description": "Groundedness is the metric that defines weather the answer follows logically from the information contained in the context or not and provides and integer score (1-5) to determine how grounded the answer is.",
        "json_schema": AnswerGroundedness
    }
}