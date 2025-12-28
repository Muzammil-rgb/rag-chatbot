import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

st.set_page_config(page_title="RAG Chatbot for CCP", layout="wide")

# ----------- Load PDFs -----------
def load_documents():
    text = ""
    if not os.path.exists("data"):
        return ""
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            pdf = PdfReader(os.path.join("data", file))
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    return text

# ----------- Build Vector DB -----------
def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectordb = FAISS.from_texts(chunks, embeddings)
    vectordb.save_local("vectorstore")

# ----------- Load Vector DB -----------
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    return FAISS.load_local("vectorstore", embeddings)

# ----------- RAG Answer -----------
def get_answer(query, vectordb):
    docs = vectordb.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    llm = OpenAI(temperature=0.2, openai_api_key=st.secrets["OPENAI_API_KEY"])

    prompt = f"""
You are a CCP RAG chatbot for university students.
Use the context to answer the question clearly.

Context:
{context}

Question: {query}

Answer:
"""
    return llm(prompt)

# ---------------- UI ----------------
st.title("📘 CCP RAG Chatbot (Streamlit Deployment)")
st.write("Upload your PDF study material in the `data` folder and click **Build Vector DB**.")

if st.button("Build Vector DB"):
    docs = load_documents()
    if docs.strip() == "":
        st.error("No PDFs found in data folder.")
    else:
        build_vectorstore(docs)
        st.success("Vector Database Created Successfully!")

vectordb = None
if os.path.exists("vectorstore"):
    vectordb = load_vectorstore()

query = st.text_input("Ask a question from your documents:")

if st.button("Ask") and vectordb:
    answer = get_answer(query, vectordb)
    st.write(answer)
