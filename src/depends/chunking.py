from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_splitter(chunk_size: int=1000, chunk_overlap: int=200, length_function: callable = len) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )