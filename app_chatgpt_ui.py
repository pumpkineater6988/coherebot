# -*- coding: utf-8 -*-
"""
Cohere AI Chatbot - Professional ChatGPT-like UI
Created for Cohere API integration
"""

import streamlit as st
import tempfile
import chardet
import pandas as pd
import os
import json
import re
import cohere
import requests
from requests.adapters import HTTPAdapter
import asyncio
import time
from typing import List, Dict, Any, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from io import StringIO
from dotenv import load_dotenv
import concurrent.futures

# ---- Load Environment Variables ----
load_dotenv()

# ---- Cohere API Configuration ----
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
COHERE_API_BASE_URL = os.environ.get("COHERE_API_BASE_URL", "https://api.cohere.ai/v1")
COHERE_MODEL = os.environ.get("COHERE_MODEL", "command-r-plus")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

# Validate API key
if not COHERE_API_KEY:
    st.error("❌ COHERE_API_KEY not found in environment variables!")
    st.info("Please create a .env file with your Cohere API key:")
    st.code("COHERE_API_KEY=your_cohere_api_key_here")
    st.stop()

# ---- Performance Optimizations ----
@st.cache_resource
def get_embeddings_model():
    """Cache the embeddings model for better performance"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_data
def process_file_content(file_content: bytes, file_type: str):
    """Cache file processing for better performance"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
    
    try:
        docs = []
        df = None
        
        if file_type == "csv":
            with open(tmp_path, 'rb') as f:
                encoding = chardet.detect(f.read())['encoding']
            df = pd.read_csv(tmp_path, encoding=encoding)
            docs = [Document(page_content=str(row.to_json())) for _, row in df.iterrows()]
        elif file_type == "xlsx":
            df = pd.read_excel(tmp_path)
            docs = [Document(page_content=str(row.to_json())) for _, row in df.iterrows()]
        elif file_type == "pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return docs, df
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

class CohereAPIClient:
    """Optimized Cohere API client with connection pooling and retry logic"""
    def __init__(self):
        self.client = cohere.Client(COHERE_API_KEY)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {COHERE_API_KEY}',
            'Content-Type': 'application/json',
            'User-Agent': 'Cohere-Chatbot/1.0'
        })
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def chat_completion(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        """Send chat completion request to Cohere API"""
        # Cohere expects a single prompt string, so concatenate messages
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        try:
            response = self.client.chat(
                model=COHERE_MODEL,
                message=prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            # Return as dict
            return {"choices": [{"message": {"content": response.text}}]}
        except Exception as e:
            raise Exception(f"Cohere API request failed: {str(e)}")

# Initialize Cohere client
cohere_client = CohereAPIClient()

# ---- Streamlit UI with ChatGPT-like Design ----
st.set_page_config(
    page_title="Cohere AI Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional ChatGPT-like CSS (Dark Theme)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main App Background */
    .stApp {
        background-color: #1e1e1e; /* Dark background */
        color: #d4d4d4;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #252526;
        border-right: 1px solid #3c3c3c;
    }

    /* Remove top padding from the main content area */
    .block-container {
        padding-top: 2rem;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1.5rem 0;
        border-bottom: 1px solid #3c3c3c;
        margin: 0;
    }

    .user-message {
        background-color: #1e1e1e;
    }

    .bot-message {
        background-color: #2d2d2d;
    }

    .message-content {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 1rem;
        font-size: 1rem;
        line-height: 1.6;
    }

    .avatar {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 32px;
        height: 32px;
        border-radius: 4px;
        color: white;
        font-weight: 600;
        margin-right: 1rem;
        flex-shrink: 0;
    }

    .user-avatar {
        background-color: #404040;
    }

    .bot-avatar {
        background-color: #3a78ff;
    }

    /* Chat Input Area */
    .st-emotion-cache-1pfl0bf {
         position: fixed;
         bottom: 0;
         left: 0;
         right: 0;
         width: 100%;
         background: linear-gradient(180deg, rgba(30, 30, 30, 0) 0%, #1e1e1e 80%);
         padding: 1.5rem 1rem 2rem 1rem;
         z-index: 1000;
    }

    /* Styling the text_area and its wrapper for chat input */
    .st-emotion-cache-1629p8f, .st-emotion-cache-1y4p8pa {
        max-width: 800px;
        margin: 0 auto;
        position: relative;
    }

    .stTextArea textarea {
        background-color: #3c3c3c;
        color: #d4d4d4;
        border: 1px solid #505050;
        border-radius: 8px;
        padding: 12px 50px 12px 15px;
        font-size: 1rem;
        min-height: 50px;
        resize: none;
        outline: none;
        transition: border-color 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: #3a78ff;
        box-shadow: 0 0 0 2px rgba(58, 120, 255, 0.2);
    }
    
    /* Send Button */
    .stButton > button[data-testid="stFormSubmitButton"] {
        position: absolute;
        right: 12px;
        bottom: 8px;
        background: #3a78ff;
        color: white;
        border: none;
        border-radius: 6px;
        width: 32px;
        height: 32px;
        cursor: pointer;
        display: flex;
        justify-content: center;
        align-items: center;
        transition: background-color 0.3s ease;
    }
    .stButton > button[data-testid="stFormSubmitButton"]:hover {
        background: #2563eb;
    }
    .stButton > button[data-testid="stFormSubmitButton"] p {
        font-size: 0; /* Hide text, we use icon */
    }
    .stButton > button[data-testid="stFormSubmitButton"]::after {
        content: '➤'; /* Send icon */
        font-size: 16px;
        line-height: 1;
    }

    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        background-color: #1e1e1e;
        border-top: 1px solid #3c3c3c;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #3c3c3c;
        color: #d4d4d4;
        border: 1px solid #505050;
        border-radius: 8px;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #3a78ff;
        box-shadow: 0 0 0 2px rgba(58, 120, 255, 0.2);
    }
    [data-testid="stChatInput"] button {
        background-color: #3a78ff;
        color: white;
        border-radius: 6px;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: #2563eb;
    }

    /* Fix for buttons in sidebar */
    section[data-testid="stSidebar"] .stButton button {
        background-color: #252526;
        color: #d4d4d4;
        border: 1px solid #3c3c3c;
        width: 100%;
        transition: all 0.3s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #3c3c3c;
        color: #ffffff;
        border-color: #3a78ff;
    }
    section[data-testid="stSidebar"] .stButton button:focus {
        outline: none;
        box-shadow: 0 0 0 2px rgba(58, 120, 255, 0.3);
    }
    
    /* Center the icons in the bottom buttons */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:last-of-type .stButton button {
        font-size: 1.25rem;
        text-align: center;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# ---- Left Sidebar ----
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0; border-bottom: 1px solid #3c3c3c; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #d4d4d4; font-size: 20px; font-weight: 600;">🤖 Cohere AI</h2>
            <p style="margin: 5px 0 0 0; color: #a0a0a0; font-size: 14px;">Professional AI Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("<h4 style='color: #d4d4d4; margin-bottom: 10px; margin-top: 20px; font-weight: 600;'>📚 Recent Chats</h4>", unsafe_allow_html=True)
    if st.session_state.chat_history:
        first_message = st.session_state.chat_history[0][1][:40] + "..." if len(st.session_state.chat_history[0][1]) > 40 else st.session_state.chat_history[0][1]
        st.markdown(f"<div style='padding: 8px 12px; background: #2d2d2d; border-radius: 6px; margin: 5px 0; font-size: 14px; color: #d4d4d4;'>{first_message}</div>", unsafe_allow_html=True)

    st.markdown("<h4 style='color: #d4d4d4; margin-bottom: 10px; margin-top: 20px; font-weight: 600;'>📎 Knowledge Base</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload documents to provide context to the chatbot.",
        type=["pdf", "csv", "xlsx"],
        label_visibility="collapsed"
    )
    if uploaded_file and not st.session_state.processing:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            st.session_state.processing = True
            try:
                file_content = uploaded_file.read()
                file_type = uploaded_file.name.split('.')[-1].lower()
                docs, _ = process_file_content(file_content, file_type)
                
                embeddings = get_embeddings_model()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = splitter.split_documents(docs)
                
                vectorstore = FAISS.from_documents(split_docs, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                st.success(f"✅ Processed {len(split_docs)} document chunks.")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
            finally:
                st.session_state.processing = False
    
    # Bottom Action Buttons
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️", key="clear", help="Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("📤", key="export", help="Export Chat to CSV"):
                if st.session_state.chat_history:
                    # Convert list of tuples to list of dicts for type-safe DataFrame creation
                    chat_data_for_df = [{"Role": role, "Message": msg} for role, msg in st.session_state.chat_history]
                    df = pd.DataFrame(chat_data_for_df)
                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False),
                        file_name="chat_history.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
        with col3:
            if st.button("🔄", key="refresh", help="Refresh Page"):
                st.rerun()

# ---- Main Chat Area ----
# Create a container for the chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='text-align: center; padding: 5rem 0;'><h1 style='color: #d4d4d4;'>Cohere AI</h1><p style='color: #a0a0a0;'>Start a conversation by typing below.</p></div>",
            unsafe_allow_html=True
        )
    else:
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-content">
                            <div style="display: flex; align-items: flex-start;">
                                <div class="avatar user-avatar">U</div>
                                <div style="flex: 1;">{message}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-content">
                            <div style="display: flex; align-items: flex-start;">
                                <div class="avatar bot-avatar">C</div>
                                <div style="flex: 1;">{message}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# ---- Chat Input ----
if user_input := st.chat_input("Message Cohere AI..."):
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("🤖 Cohere is thinking..."):
        try:
            context = ""
            if st.session_state.retriever:
                docs = st.session_state.retriever.get_relevant_documents(user_input)
                context = "\n\n".join([doc.page_content for doc in docs])

            messages = [{"role": "system", "content": "You are a helpful and intelligent AI assistant."}]
            if context:
                messages.append({"role": "user", "content": f"Context:\n{context}\n\nUser question: {user_input}"})
            else:
                 messages.append({"role": "user", "content": user_input})

            # Add limited history
            for role, msg in st.session_state.chat_history[-4:-1]: # last 2 exchanges
                messages.append({"role": "user" if role == "user" else "assistant", "content": msg})

            response = cohere_client.chat_completion(messages)
            answer = response['choices'][0]['message']['content']
            st.session_state.chat_history.append(("assistant", answer))

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.chat_history.append(("assistant", error_msg))

    st.rerun()

# ---- Footer ----
st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 20px; font-size: 12px;">
        <p>🚀 Powered by <strong>Cohere AI</strong> | Professional AI Assistant</p>
    </div>
""", unsafe_allow_html=True)