import streamlit as st
from loaders.pdf_loader import load_pdfs
from utils.splitter import split_documents
from vectorstore.qdrant_setup import get_vectorstore
from llm.synthesis_chain import get_response_chain

st.title("ShayariDB Poetry Retrieval")

pdf_files = [
    "Sahayari/allama_iqbal.pdf",
    # Add more PDF paths as needed
]

# Load and prepare documents
documents = load_pdfs(pdf_files)
split_docs = split_documents(documents)
vectorstore = get_vectorstore(split_docs)

# UI for query
query = st.text_input("Enter your query to retrieve a shayari:")
if query:
    results = vectorstore.similarity_search(query, k=5)
    # st.write("Results:")
    # for result in results:
        # st.write(result.page_content)

    # LLM response
    chain = get_response_chain(results)
    response = chain.invoke(query)
    st.write("Synthesis Response:")
    st.write(response)

