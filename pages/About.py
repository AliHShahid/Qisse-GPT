import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://t4.ftcdn.net/jpg/04/87/89/71/240_F_487897177_ZENJsC7013PpjX5qDcr657jEQKybrT7E.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 About Page")
st.write("RAIVA is an AI-powered virtual assistant designed to help you with a variety of tasks.")
st.write("Whether you need assistance with scheduling, information retrieval, or just a friendly chat, RAIVA is here to help.")
