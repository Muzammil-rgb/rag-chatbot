import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from utils import load_docs, create_vectorstore

st.set_page_config(page_title="RAG Chatbot")
st.title("📚 RAG-based Chatbot")

# Load documents
docs = load_docs()
vectorstore = create_vectorstore(docs)

# Initialize language model
llm = OpenAI(temperature=0)

# Initialize RAG chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask me anything:")

if query:
    result = qa.run(query)
    st.session_state.chat_history.append((query, result))
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
