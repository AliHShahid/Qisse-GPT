from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(pdf_files):
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())
    return documents
