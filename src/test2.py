from langchain.prompts import PromptTemplate
from depends.model import llm_anton_llama2

# Step 1: Define the PromptTemplate
template = "Write a short story about a {animal} who {action}."
prompt_template = PromptTemplate(template=template)

# Step 2: Format the Prompt
formatted_prompt = prompt_template.format(animal="cat", action="finds a magical item")

# Step 3: Generate Responses
response = llm_anton_llama2.generate([formatted_prompt])
print(response)