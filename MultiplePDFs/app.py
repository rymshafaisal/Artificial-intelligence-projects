import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- CONFIG ----------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Chat with PDFs", page_icon="📚", layout="wide")

# ----------------- CUSTOM STYLE -----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }

    /* Title */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    /* Input box */
    textarea, input[type="text"] {
        background: rgba(255,255,255,0.08) !important;
        color: #f1f5f9 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        padding: 12px !important;
    }
    textarea::placeholder, input::placeholder {
        color: #94a3b8 !important;
    }

    /* File uploader */
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #cbd5e1 !important;
        padding: 20px !important;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none;
        font-weight: 600;
        padding: 10px 20px;
        transition: 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }

    /* Chat response box */
    .chat-bubble {
        background: rgba(255,255,255,0.08);
        padding: 15px 20px;
        border-radius: 14px;
        margin-bottom: 12px;
        line-height: 1.6;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .user-question {
        background: rgba(59,130,246,0.15);
        text-align: right;
        color: #93c5fd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- HELPERS -----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If answer is not in context just say "Answer is not available in context".
    Do not make up an answer.

    Context: \n {context}\n
    Question: \n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.markdown(f"<div class='chat-bubble user-question'><b>You:</b> {user_question}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble'><b>Gemini:</b> {response['output_text']}</div>", unsafe_allow_html=True)

# ----------------- LAYOUT -----------------
st.title("📚 Chat with Multiple PDFs using Gemini")
st.markdown("Upload multiple PDFs, process them into a knowledge base, and ask questions interactively.")

# Main input
user_question = st.text_input("💬 Ask a Question from your documents", placeholder="Type your question here...")

if user_question:
    user_input(user_question)

# Sidebar
with st.sidebar:
    st.title("📂 Menu")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])

    if st.button("⚡ Submit & Process"):
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("✅ PDFs processed successfully! Now ask your questions above.")
