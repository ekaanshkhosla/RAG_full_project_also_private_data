import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def load_documents():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(current_dir, '../../data/pdfs')
    pdf_files = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir) if file.endswith(".pdf")]
    
    urls = [
        "https://www.math-datascience.nat.fau.de/im-studium/masterstudiengaenge/master-data-science/",
        "https://www.fau.eu/studiengang/data-science-bsc/",
        "https://www.fau.eu/studiengang/data-science-msc/"
    ]
    
    all_doc_splits = []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP)
    
    for pdf_file in pdf_files:
        pdf_loader = PyPDFLoader(pdf_file)
        documents = pdf_loader.load()
        doc_splits = text_splitter.split_documents(documents)
        all_doc_splits.extend(doc_splits)
    
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    web_doc_splits = text_splitter.split_documents(docs_list)
    all_doc_splits.extend(web_doc_splits)
    
    return all_doc_splits
