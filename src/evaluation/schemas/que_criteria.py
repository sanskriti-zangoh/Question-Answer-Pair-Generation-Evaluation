from pydantic import BaseModel, Field

# Fluency: Finally, we use the fluency metric that leverages large language models, such as GPT-4, to assess
# fluency and coherence. By preparing a prompt that instructs the AI to rate a given question on a scale
# of 1 to 5 and provide an explanation, the model can generate insightful scores for each question. After
# appending the generated question to the prompt, submitting it to GPT-4, and parsing the response to extract the
# fluency score and explanation, the results can be stored for further analysis. This approach effectively utilizes
# GPT-4’s language understanding capabilities to assess the quality of generated questions, aiding in refining
# question-generation models and selecting the best questions for specific applications. An example is provided
# in Table 5.

class QuestionFluency(BaseModel):
    score: int = Field(ge=0, le=5)
    reasoning: str

class QuestionCriteria(BaseModel):
    fluency: QuestionFluency = Field(description="Fluency is a metric which rates a given question on a scale of 1 to 5 to assess its fluency and coherence.")


question_criteria = {
    "question_fluency": {
        "description": "Fluency is a metric which rates a given question on a scale of 1 to 5 to assess its fluency and coherence.",
        "json_schema": QuestionFluency
    }
}

