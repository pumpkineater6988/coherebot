# -*- coding: utf-8 -*-
"""
Cohere AI Chatbot - Optimized for Performance
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

def test_api_connection():
    """Test Cohere API connection and show results"""
    try:
        with st.spinner("Testing API connection..."):
            # Test basic authentication
            headers = {
                'Authorization': f'Bearer {COHERE_API_KEY}',
                'Content-Type': 'application/json',
                'User-Agent': 'Cohere-Chatbot/1.0'
            }
            # Test models endpoint (Cohere does not have a public models endpoint, so test chat)
            payload = {
                "model": COHERE_MODEL,
                "message": "Hello",
                "max_tokens": 10
            }
            response = requests.post(
                f"{COHERE_API_BASE_URL}/chat",
                headers=headers,
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                st.success("✅ API connection successful!")
                st.info("Chat test successful!")
            elif response.status_code == 401:
                st.error("❌ 401 Unauthorized - Invalid API key")
            elif response.status_code == 403:
                st.error("❌ 403 Forbidden - Authentication failed")
            else:
                st.error(f"❌ API test failed: {response.status_code}")
                st.text(f"Response: {response.text}")
    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")
        st.info("Check your internet connection and try again.")

# ---- Streamlit UI with Performance Optimizations ----
st.set_page_config(
    page_title="Cohere AI Chatbot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optimized CSS for better performance
st.markdown("""
    <style>
    .chat-message {
        padding: 12px 16px;
        border-radius: 18px;
        margin-bottom: 12px;
        max-width: 85%;
        word-break: break-word;
        font-size: 1.1em;
        line-height: 1.4;
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        align-self: flex-end;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        align-self: flex-start;
        margin-right: auto;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
        padding: 16px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Cohere AI Chatbot")
st.markdown("### Powered by Cohere AI - Optimized for Performance")

# ---- Session State with Performance Optimizations ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# ---- Sidebar with File Upload ----
with st.sidebar:
    st.header("📁 Upload Knowledge Base")
    st.markdown("Upload documents to enhance chatbot responses")
    
    uploaded_file = st.file_uploader(
        "Choose PDF, CSV, or Excel file", 
        type=["pdf", "csv", "xlsx"],
        help="Upload documents to provide context for the chatbot"
    )
    
    if uploaded_file:
        if st.button("🚀 Process & Embed", type="primary"):
            if not st.session_state.processing:
                st.session_state.processing = True
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📖 Reading file...")
                    progress_bar.progress(25)
                    
                    file_content = uploaded_file.read()
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    status_text.text("🔧 Processing content...")
                    progress_bar.progress(50)
                    
                    docs, df = process_file_content(file_content, file_type)
                    st.session_state.dataframe = df
                    
                    status_text.text("🧠 Creating embeddings...")
                    progress_bar.progress(75)
                    
                    # Use cached embeddings model
                    embeddings = get_embeddings_model()
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200,
                        length_function=len
                    )
                    split_docs = splitter.split_documents(docs)
                    
                    status_text.text("💾 Building vector store...")
                    progress_bar.progress(90)
                    
                    vectorstore = FAISS.from_documents(split_docs, embeddings)
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Ready!")
                    st.success(f"📚 Processed {len(split_docs)} chunks from {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    st.session_state.processing = False
                    progress_bar.empty()
                    status_text.empty()

# ---- Main Chat Interface ----
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history with performance optimization
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.markdown(f'<div class="chat-message user-message">👤 {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">🤖 {message}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input with performance optimizations
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "💬 Ask me anything...", 
            key="user_input",
            height=100,
            placeholder="Type your message here..."
        )
        
        submitted = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)
    
    # Action buttons outside the form
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("📤 Export Chat", use_container_width=True) and st.session_state.chat_history:
            try:
                chat_data = []
                for role, msg in st.session_state.chat_history:
                    chat_data.append({"Role": role, "Message": msg})
                csv_data = pd.DataFrame(chat_data)
                csv_buffer = StringIO()
                csv_data.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download Chat History",
                    csv_buffer.getvalue(),
                    file_name=f"cohere_chat_history_{int(time.time())}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Export error: {e}")
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

with col2:
    st.markdown("### 📊 Status")
    if st.session_state.retriever:
        st.success("✅ Knowledge Base Loaded")
        if st.session_state.dataframe is not None:
            st.info(f"📊 Data: {len(st.session_state.dataframe)} rows")
    else:
        st.warning("⚠️ No Knowledge Base")
    
    st.markdown("### ⚙️ Settings")
    st.metric("Messages", len(st.session_state.chat_history))
    st.metric("Model", COHERE_MODEL)
    
    st.markdown("### 🔧 API Configuration")
    st.info(f"**Model:** {COHERE_MODEL}")
    st.info(f"**Endpoint:** {COHERE_API_BASE_URL}")
    st.info(f"**API Key:** {'✅ Set' if COHERE_API_KEY else '❌ Missing'}")
    if COHERE_API_KEY:
        key_preview = COHERE_API_KEY[:10] + "..." if len(COHERE_API_KEY) > 10 else COHERE_API_KEY
        st.info(f"**Key Preview:** {key_preview}")
    if st.button("🔍 Test API Connection"):
        test_api_connection()

# ---- Handle Chat Submission ----
if submitted and user_input and not st.session_state.processing:
    st.session_state.processing = True
    try:
        # Get context from knowledge base
        context = ""
        if st.session_state.retriever:
            with st.spinner("🔍 Searching knowledge base..."):
                docs = st.session_state.retriever.get_relevant_documents(user_input)
                context = "\n\n".join([doc.page_content for doc in docs])
        # Prepare messages with context
        messages = [
            {
                "role": "system", 
                "content": "You are Cohere, a helpful and intelligent AI assistant. Provide accurate, helpful, and engaging responses. If context is provided, use it to enhance your answers."
            }
        ]
        if context:
            messages.append({
                "role": "user", 
                "content": f"Context from knowledge base:\n{context}\n\nUser question: {user_input}"
            })
        else:
            messages.append({"role": "user", "content": user_input})
        for role, msg in st.session_state.chat_history[-8:]:
            messages.append({"role": role, "content": msg})
        with st.spinner("🤖 Cohere is thinking..."):
            response = cohere_client.chat_completion(messages)
            if 'choices' in response and len(response['choices']) > 0:
                answer = response['choices'][0]['message']['content']
            else:
                answer = "Sorry, I couldn't generate a response at the moment."
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", answer))
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", error_msg))
        st.error(error_msg)
    finally:
        st.session_state.processing = False
        st.rerun()

# ---- Footer ----
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 Powered by <strong>Cohere AI</strong> | Optimized for Performance</p>
        <p>💡 Upload documents to enhance responses | 🔄 Real-time chat with no lag</p>
    </div>
    """, 
    unsafe_allow_html=True
) 