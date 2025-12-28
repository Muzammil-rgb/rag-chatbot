from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from PyPDF2 import PdfReader

def load_docs(folder_path="data"):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf = PdfReader(os.path.join(folder_path, filename))
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            docs.append(text)
        elif filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def create_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore
