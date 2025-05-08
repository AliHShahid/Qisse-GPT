import streamlit as st
from loaders.pdf_loader import load_pdfs
from utils.splitter import split_documents
from vectorstore.qdrant_setup import get_vectorstore
from llm.synthesis_chain import get_response_chain

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://mir-s3-cdn-cf.behance.net/project_modules/source/f39a90223734615.67fe56e31794b.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Qisse-GPT: Shayari Retrieval and Synthesis")
st.write("Welcome to the Shayari Retrieval and Synthesis application!")

pdf_files = [
    "Sahayari/ahmad_faraz.pdf",
    "Sahayari/allama_iqbal.pdf",
    "Sahayari/faiz_ahmad_faiz.pdf",
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

