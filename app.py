import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from utils import load_docs, create_vectorstore

st.set_page_config(page_title="RAG Chatbot")
st.title("📚 RAG-based Chatbot")

# Load documents and create vector store
docs = load_docs()
vectorstore = create_vectorstore(docs)

# Initialize chat model
llm = ChatOpenAI(temperature=0)
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask me anything:")

if query:
    result = qa({"question": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((query, result["answer"]))
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
