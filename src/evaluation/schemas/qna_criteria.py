from pydantic import BaseModel, Field

# Relevance: To measure the informativeness of a generated Q&A pair from the perspective of a farmer, we
# employ Large Language Model (LLMs) - namely GPT-4 - to rate the question on a scale of 1 to 5, with 1 being
# a question that would be asked by a farmer and 5 a question that would not, given the context. This metric
# ensures that the generated content is relevant and accurate to the target audience, considering all provided
# information. An example is provided in Table 2.

class UserRelevancy(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str

# Global Relevance: To measure the informativeness of a generated Q&A pair from the perspective of a farmer
# without considering any context, we employ Large Language Model (LLMs) - namely GPT-4 - to rate the
# question on a scale of 1 to 5, with 1 being a question that would be asked by a farmer and 5 a question that
# would not. This metric ensures that the generated content is relevant and accurate to the target audience. An
# example is provided in Table 3.

class GlobalRelevancy(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


class QnACriteria(BaseModel):
    relevancy: UserRelevancy = Field(description="Relevance metric is to measure the informativeness of a generated Q&A pair from the perspective of a user. The metric rates from 1 to 5, with 1 being a question that would be asked by the target user and 5 a question that would not, given the context. This metric ensures that the generated content is relevant and accurate to the target audience, considering all provided information.")
    global_relevancy: GlobalRelevancy = Field(description="Global Relevance metric is to measure the informativeness of a generated Q&A pair from the perspective of a user without considering any context. This rates from 1 to 5, with 1 being a question that would be asked by the target user and 5 a question that would not. This metric ensures that the generated content is relevant and accurate to the target audience.")


question_answer_criteria = {
    "user_relevancy": {
        "description": "Relevance metric is to measure the informativeness of a generated Q&A pair from the perspective of a user. The metric rates from 1 to 5, with 1 being a question that would be asked by the target user and 5 a question that would not, given the context. This metric ensures that the generated content is relevant and accurate to the target audience, considering all provided information.",
        "json_schema": UserRelevancy
    },
    "global_relevancy": {
        "description": "Global Relevance metric is to measure the informativeness of a generated Q&A pair from the perspective of a user without considering any context. This rates from 1 to 5, with 1 being a question that would be asked by the target user and 5 a question that would not. This metric ensures that the generated content is relevant and accurate to the target audience.",
        "json_schema": GlobalRelevancy
    }
}