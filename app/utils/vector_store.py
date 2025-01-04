from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_nomic import NomicEmbeddings
from dotenv import load_dotenv
import pickle
import faiss
from langchain_community.vectorstores import Chroma
from app.config import (
    EMBEDDING,
)


# Load the keys
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '../../keys/.env')
load_dotenv(dotenv_path=dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
nomic_api_key = os.getenv("NOMIC_API_KEY")


def create_vector_store(doc_splits):
    
    # Create a FAISS vector store
    vectorstore = FAISS.from_documents(
        documents=doc_splits, 
        embedding=EMBEDDING
    )
    
    # Save the FAISS vector store to disk
    vectorstore.save_local("faiss_index")
    
    # Return the retriever object for the vector store
    return vectorstore.as_retriever()

def load_vector_store():

    # Load the FAISS vector store from disk
    vectorstore = FAISS.load_local("faiss_index", EMBEDDING, allow_dangerous_deserialization=True)

    return vectorstore.as_retriever()