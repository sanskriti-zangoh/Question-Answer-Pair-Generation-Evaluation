from evaluation.schemas.ans_criteria import AnswerCriteria, answer_criteria
from evaluation.schemas.qna_criteria import question_answer_criteria, QnACriteria
from evaluation.schemas.que_criteria import question_criteria, QuestionCriteria

from typing import Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, ListOutputParser, StrOutputParser

def get_ans_evaluation_prompt_parser() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=AnswerCriteria
    )
    prompt = PromptTemplate(
        template="You are given question, answer and context from which answer was formed. Your job is to impartially evaluate the given answer and output scores with proper detailed reasoning in only and only in json format.\n\nQuestion: {question}\n\nAnswer: {answer}\n\nAnswerContext: {answer_context} \n\n{format_instructions}",
        input_variables=["question", "answer", "answer_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

def get_que_evaluation_prompt_parser() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=QuestionCriteria
    )
    prompt = PromptTemplate(
        template="You are given the question and the context from which the question was formed. Your job is to impartially evaluate the given question and output scores with proper detailed reasoning in only and only in json format.\n\nQuestion: {question}\n\nQuestionContext: {question_context}\n\n{format_instructions}",
        input_variables=["question", "question_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

def get_qna_evaluation_prompt_parser() -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=QnACriteria
    )
    prompt = PromptTemplate(
        template="You are given question, context from which question was formed, answer and the context from which answer was formed. Your job is to impartially evaluate the given question and answer pair and output scores with proper detailed reasoning in only and only in json format.\n\nQuestion: {question}\n\nQuestionContext: {question_context}\n\nAnswer: {answer}\n\nAnswerContext: {answer_context}\n\n{format_instructions}",
        input_variables=["question", "question_context", "answer", "answer_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

def get_ans_criteria_evaluation_prompt_parser(metric: str) -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=answer_criteria[metric]['json_schema']
    )
    prompt = PromptTemplate(
        template="You are given question, answer and context from which answer was formed. Your job is to impartially evaluate the given answer on the given metric and output scores with proper detailed reasoning in only and only json format.\n\nQuestion: {question}\n\nAnswer: {answer}\n\nAnswerContext: {answer_context}\n\nMetric: {metric_name}\tDescription: {metric}\n\n{format_instructions}",
        input_variables=["question", "answer", "answer_context"],
        partial_variables={"format_instructions": parser.get_format_instructions(), 'metric': answer_criteria[metric]['description'], 'metric_name': metric},
    )
    return prompt, parser

def get_que_criteria_evaluation_prompt_parser(metric: str) -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=question_criteria[metric]['json_schema']
    )
    prompt = PromptTemplate(
        template="You are given the question and the context from which the question was formed. Your job is to impartially evaluate the given question on the given metric and output scores with proper detailed reasoning in only and only json format.\n\nQuestion: {question}\n\nQuestionContext: {question_context}\n\nMetric: {metric_name}\tDescription: {metric}\n\n{format_instructions}",
        input_variables=["question", "question_context"],
        partial_variables={"format_instructions": parser.get_format_instructions(), 'metric': question_criteria[metric]['description'], 'metric_name': metric},
    )
    return prompt, parser

def get_qna_criteria_evaluation_prompt_parser(metric: str) -> Tuple[PromptTemplate, JsonOutputParser]:
    parser = JsonOutputParser(
        pydantic_object=question_answer_criteria[metric]['json_schema']
    )
    prompt = PromptTemplate(
        template="You are given question, context from which question was formed, answer and the context from which answer was formed. Your job is to impartially evaluate the given question and answer pair on the given metric and output scores with proper detailed reasoning in only and only json format.\n\nQuestion: {question}\n\nQuestionContext: {question_context}\n\nAnswer: {answer}\n\nAnswerContext: {answer_context}\n\nMetric: {metric_name}\tDescription: {metric}\n\n{format_instructions}",
        input_variables=["question", "question_context", "answer", "answer_context"],
        partial_variables={"format_instructions": parser.get_format_instructions(), 'metric': question_answer_criteria[metric]['description'], 'metric_name': metric},
    )
    return prompt, parser