import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from app.config import VECTORSTORE_PATH
from app.llm_model import CTransformerLLM

# Set up Streamlit page
st.set_page_config(page_title="HR Policy Chatbot")
st.markdown("<h1 style='text-align: center;'>💼 HR Policy Chatbot</h1>", unsafe_allow_html=True)

# Inject custom CSS for left-right chat layout
st.markdown("""
    <style>
        .user-msg {
            background-color: #DCF8C6;
            padding: 10px 15px;
            border-radius: 20px;
            max-width: 70%;
            margin-left: auto;
            margin-right: 0;
            margin-bottom: 10px;
        }
        .bot-msg {
            background-color: #F1F0F0;
            padding: 10px 15px;
            border-radius: 20px;
            max-width: 70%;
            margin-right: auto;
            margin-left: 0;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_chain():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = CTransformerLLM()
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# Initialize chain
chain = init_chain()

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Handle user input
user_input = st.chat_input("Ask about HR policies...")
if user_input:
    response = chain.run(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# Display chat bubbles
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='user-msg'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{message}</div>", unsafe_allow_html=True)
