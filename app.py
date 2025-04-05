import streamlit as st
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os

# Load documents
pdf_files = [
    # "Sahayari/ahmad_faraz.pdf",
    "Sahayari/allama_iqbal.pdf",
    # "Sahayari/faiz_ahmad_faiz.pdf",
    # "Sahayari/haider_ali_atish.pdf",
    # "Sahayari/jaun_elia.pdf",
    # "Sahayari/mir_taqi_mir.pdf",
    # "Sahayari/mirza_ghalib.pdf",
    # "Sahayari/muneer_niyazi.pdf",
    # "Sahayari/nazeer_akbarabadi.pdf",
    # "Sahayari/nida_fazli.pdf",
    # "Sahayari/parveen_shakir.pdf",
    # "Sahayari/qateel_shifai.pdf",
    # "Sahayari/riyaz_khairabadi.pdf",
    # "Sahayari/siraj_aurangabadi.pdf",
    # "Sahayari/zafar_iqbal.pdf"
]

documents = []
for pdf_file in pdf_files:
    pdf_loader = PyPDFLoader(pdf_file)
    documents += pdf_loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=30)
split_docs = text_splitter.split_documents(documents)

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Qdrant VectorStore
qdrant_url = "Ypour Qdrant URL"
qdrant_key = "Your Qdrant API Key"
collection_name = "Poetry DB VSCode"
qdrant = QdrantVectorStore.from_documents(
    split_docs,
    embeddings,
    url=qdrant_url,
    api_key=qdrant_key,
    collection_name=collection_name
)

# Streamlit UI
st.title("ShayariDB Poetry Retrieval")
query = st.text_input("Enter your query to retrieve a shayari:")

if query:
    results = qdrant.similarity_search(query, k=5)
    st.write("Results:")
    for result in results:
        st.write(result.page_content)

    # Synthesis using LLM
    os.environ["GROQ_API_KEY"] = "Your Groq API Key"
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = PromptTemplate(template=template)

    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    chain = (
        RunnableLambda(lambda x: {"context": format_docs(results), "question": x})
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query)
    st.write("Synthesis Response:")
    st.write(response)
