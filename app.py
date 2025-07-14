import os
import streamlit as st
import pandas as pd
import tempfile
import chardet
from dotenv import load_dotenv
from io import StringIO
import cohere
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit.components.v1 as components

# ---- Load .env ----
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_MODEL = os.getenv("COHERE_MODEL", "command-r-plus")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

if not COHERE_API_KEY:
    st.error("❌ COHERE_API_KEY not found in environment variables.")
    st.stop()

# ---- Cohere Client ----
class CohereAPIClient:
    def __init__(self):
        self.client = cohere.Client(COHERE_API_KEY)

    def chat_completion(self, messages):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = self.client.chat(
            model=COHERE_MODEL,
            message=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        return response.text

client = CohereAPIClient()

# ---- Helpers ----
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data
def process_file(content, file_type):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    docs = []
    if file_type == "pdf":
        docs = PyPDFLoader(tmp_path).load()
    elif file_type == "csv":
        with open(tmp_path, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
        df = pd.read_csv(tmp_path, encoding=encoding)
        docs = [Document(page_content=str(row.to_json())) for _, row in df.iterrows()]
    elif file_type == "xlsx":
        df = pd.read_excel(tmp_path)
        docs = [Document(page_content=str(row.to_json())) for _, row in df.iterrows()]
    os.unlink(tmp_path)
    return docs

# ---- Session State ----
st.set_page_config(page_title="Cohere ChatGPT UI", layout="wide")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("show_typing", False)
st.session_state.setdefault("retriever", None)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("""
        <h2 style='color:#d4d4d4; margin-bottom: 0.5rem;'>🤖 Cohere Chat</h2>
        <p style='color:#a0a0a0; margin-top:0; font-size:14px;'>ChatGPT-like AI Assistant</p>
        <hr style='border:1px solid #333;'>
    """, unsafe_allow_html=True)
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    file = st.file_uploader("Upload PDF/CSV/XLSX for KB", type=["pdf", "csv", "xlsx"])
    if file:
        docs = process_file(file.read(), file.name.split('.')[-1].lower())
        vecs = FAISS.from_documents(docs, get_embeddings_model())
        st.session_state.retriever = vecs.as_retriever(search_kwargs={"k": 5})
        st.success("✅ Knowledge base loaded.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("📤 Export Chat", use_container_width=True):
        chat_data_for_df = [{"role": role, "message": msg} for role, msg in st.session_state.chat_history]
        df = pd.DataFrame(chat_data_for_df)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="chat_history.csv")

# ---- Custom CSS & JS ----
st.markdown("""
<style>
body, .stApp { background: #18181b !important; }
.stApp { color: #d4d4d4; }
section[data-testid="stSidebar"] { background: #23232a !important; border-right: 1px solid #333; min-width: 320px !important; }
hr { border-color: #333; }

.chat-window { max-height: 75vh; overflow-y: auto; padding: 20px; }
.chat-message { display: flex; margin-bottom: 14px; }
.chat-avatar { font-size: 22px; margin-right: 12px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; border-radius: 50%; background: #23232a; }
.user-bubble { background: #3a78ff; color: white; margin-left: auto; border-bottom-right-radius: 4px; }
.bot-bubble { background: #2d2d2d; color: white; margin-right: auto; border-bottom-left-radius: 4px; }
.chat-bubble { padding: 14px 18px; border-radius: 18px; max-width: 75%; line-height: 1.6; word-wrap: break-word; font-size: 1.08rem; box-shadow: 0 2px 8px #0002; }

/* Sidebar Button Styling */
.stButton>button {
  background: #23232a !important;
  color: #fff !important;
  border-radius: 8px;
  border: 1px solid #444 !important;
  padding: 0.7rem 1.2rem;
  font-weight: 600;
  font-size: 1rem;
  margin-left: 0;
  margin-bottom: 16px;
  transition: background 0.2s, color 0.2s;
}
.stButton>button:hover {
  background: #3a78ff !important;
  color: #fff !important;
  border-color: #3a78ff !important;
}

.chat-input-bar { position: fixed; bottom: 0; left: 0; right: 0; padding: 18px 0 18px 0; background: linear-gradient(180deg, rgba(24,24,27,0.7) 0%, #18181b 90%); border-top: 1px solid #333; z-index: 999; }
.stTextInput>div>div>input { background-color: #23232a !important; color: #fff !important; border-radius: 8px; padding: 14px; font-size: 1.08rem; }
.stButton>button { background: #3a78ff; color: #fff; border-radius: 8px; border: none; padding: 0.7rem 1.2rem; font-weight: 600; font-size: 1rem; margin-left: 10px; }
.stButton>button:hover { background: #2563eb; }

@media (max-width: 700px) {
  .chat-window { padding: 8px; }
  section[data-testid="stSidebar"] { min-width: 100vw !important; }
  .chat-bubble { font-size: 1rem; }
}
</style>
<script>
window.addEventListener('load', function() {
  window.scrollTo(0, document.body.scrollHeight);
});
</script>
""", unsafe_allow_html=True)

# ---- Chat History ----
st.markdown('<div class="chat-window">', unsafe_allow_html=True)
for role, message in st.session_state.chat_history:
    avatar = "🧑" if role == "user" else "🤖"
    bubble_class = "user-bubble" if role == "user" else "bot-bubble"
    st.markdown(f"""
    <div class="chat-message">
        <div class="chat-avatar">{avatar}</div>
        <div class="chat-bubble {bubble_class}">{message}</div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.show_typing:
    st.markdown("""
    <div class="chat-message">
        <div class="chat-avatar">🤖</div>
        <div class="chat-bubble bot-bubble"><em>Bot is typing...</em></div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---- Input Bar (Fixed) ----
# Clear input if flagged (before widget is rendered)
if st.session_state.get("clear_input", False):
    st.session_state["chat_input"] = ""
    st.session_state["clear_input"] = False

st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)
col1, col2 = st.columns([10, 1])
with col1:
    user_input = st.text_input("Type your message...", key="chat_input", label_visibility="collapsed")
with col2:
    send = st.button("➤", key="send_btn", help="Send", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---- Chat Logic ----
if (send or (user_input and user_input != st.session_state.get("_last_input"))) and user_input.strip():
    st.session_state.chat_history.append(("user", user_input.strip()))
    st.session_state.show_typing = True
    st.session_state["_last_input"] = user_input
    st.session_state["clear_input"] = True  # Set flag to clear input on next run
    st.rerun()

if st.session_state.show_typing:
    context = ""
    if st.session_state.retriever:
        docs = st.session_state.retriever.get_relevant_documents(st.session_state.get("_last_input", ""))
        context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\nUser: {st.session_state.get('_last_input', '')}" if context else st.session_state.get('_last_input', '')
    messages = [{"role": "user", "content": prompt}]
    reply = client.chat_completion(messages)
    st.session_state.chat_history.append(("assistant", reply))
    st.session_state.show_typing = False
    st.rerun() 
