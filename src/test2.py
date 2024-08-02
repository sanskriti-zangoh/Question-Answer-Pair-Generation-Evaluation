from depends.model import llm_anton_local_llama3

chat_completion = llm_anton_local_llama3.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What is a contract farming? Explain in detail",
        }
    ],
    model="llama3",
)

print(chat_completion.choices[0].message.content)
