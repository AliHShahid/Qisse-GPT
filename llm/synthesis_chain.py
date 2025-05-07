import os
from langchain.schema.runnable import RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config import GROQ_API_KEY
from .prompt import get_prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_response_chain(results):
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    prompt = get_prompt()
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    chain = (
        RunnableLambda(lambda x: {"context": format_docs(results), "question": x})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
