from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=120, chunk_overlap=30):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
