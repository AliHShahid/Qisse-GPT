from langchain_core.prompts import PromptTemplate

def get_prompt():
    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    return PromptTemplate(template=template)
