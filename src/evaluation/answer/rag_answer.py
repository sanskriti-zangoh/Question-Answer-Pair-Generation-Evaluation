import giskard
import giskard.llm
import pandas as pd
import os
from giskard.llm.client.openai import OpenAIClient
from langchain_core.prompts import PromptTemplate

def model_predict_rag_answer(llm_chain, df: pd.DataFrame, prompt: PromptTemplate):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and must return a list of the outputs (one for each row).
    """
    return [llm_chain.invoke({"question": question, "rag_context": rag_context}) for question, rag_context in zip(df["question"], df["rag_context"])]

def get_giskard_model(model_type:str="text_generation", name:str="Agriculture", description:str="This model answers any question about agriculture based on Vikaspedia", feature_names: list=["question", "rag_context"]) -> giskard.Model:
    giskard_model = giskard.Model(
        model=model_predict_rag_answer,
        model_type="text_generation",
        name="Climate Change Question Answering",
        description="This model answers any question about climate change based on IPCC reports",
        feature_names=["question", "rag_context"],
    )

def get_giskard_dataset(df: pd.DataFrame) -> giskard.Dataset:
    giskard_dataset = giskard.Dataset(df, target=None)
    return giskard_dataset

def prediction_function(inputs, llm_chain):
    df = pd.DataFrame(inputs)
    return model_predict_rag_answer(llm_chain, df)
