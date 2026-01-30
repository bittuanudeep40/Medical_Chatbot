import streamlit as st
import os
import io
import tempfile
import uuid
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.helper import download_hugging_face_embeddings
from src.prompt import *

# --- Configuration & Setup ---
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(page_title="Medical Chatbot")
st.title("Medical Chatbot")

# Initialize Embeddings & LLM
@st.cache_resource
def load_models():
    embeddings = download_hugging_face_embeddings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        temperature=0.4, 
        max_output_tokens=500
    )
    return embeddings, llm

embeddings, llm = load_models()

system_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("Upload Medical Report")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing..."):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            
            if text.strip():
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.session_state.vector_store = vector_store
                st.success("Analysis complete! You can now ask questions.")
            else:
                st.error("Could not extract text from PDF.")

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.vector_store is not None:
            retriever = st.session_state.vector_store.as_retriever()
            question_answer_chain = create_stuff_documents_chain(llm, system_prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": prompt})
            answer = response.get("answer", "I couldn't find an answer.")
        else:
            answer = "Please upload a medical report PDF in the sidebar first."
            
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
