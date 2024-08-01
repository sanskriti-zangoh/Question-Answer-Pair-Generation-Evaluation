from pydantic import BaseModel, Field

# Coverage: To gauge whether the generated answers can be directly extracted from the provided context and
# how well the model avoids hallucinating, we use LLMs to rate the answerability of each Q&A pair on a scale
# of 1 to 5. The prompt was the following: “Your task is to rate from 1 to 5 if the answer can be extracted from
# the context and the question”. A higher score indicates that the answer can be more reliably extracted from
# the context, ensuring the model’s output is grounded in the available information. An example is provided in
# Table 4.

class AnswerCoverage(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str

# Relevance: Relevance measures how well the answer addresses the main aspects of the question based on
# the context. The metric rates from 1 to 5, where 5 means the answer has perfect relevance. An example is
# provided in Table 7.

class AnswerRelevancy(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str

# Groundedness: The metric defines weather the answer follows logically from the information contained in
# the context or not and provides and integer score to determine how grounded the answer is. An example is
# provided in Table 8

class AnswerGroundedness(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str