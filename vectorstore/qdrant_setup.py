from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_MODEL

def get_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    qdrant = QdrantVectorStore.from_documents(
        documents,
        embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )
    return qdrant
