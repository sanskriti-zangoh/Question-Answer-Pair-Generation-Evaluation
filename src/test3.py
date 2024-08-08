from depends.model import llm_anton_llama3, llm_anton_llama3_1

print(llm_anton_llama3.invoke("How birds fly?"))

chat_completion = llm_anton_llama3_1.chat.completions.create(
messages=[
    {
        "role": "user",
        "content": f"How birds fly?",
    }
],
model="meta-llama/Meta-Llama-3.1-8B-Instruct",
)

response = chat_completion.choices[0].message.content
print("RESPONSE: ", response)