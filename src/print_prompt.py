from depends.prompt import get_qna_prompt_parser_from_chunks, get_qna_prompt_parser_from_chunks2, get_qna_prompt_parser_from_chunks3, get_que_prompt_parser_from_chunks

prompt, parser = get_que_prompt_parser_from_chunks()
print(prompt)