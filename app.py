import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from src.prompt import *
import os
import io
from PyPDF2 import PdfReader

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Medical Chatbot", layout="wide")

# --- LOAD ENVIRONMENT VARIABLES AND API KEY ---
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
# Note: For Streamlit Cloud, set this as a secret in the app settings.
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- CACHED RESOURCES ---
# Cache the LLM and embeddings models to prevent reloading on every interaction.
@st.cache_resource
def load_llm():
    """Loads the Google Generative AI model."""
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4, max_output_tokens=1000)

@st.cache_resource
def load_embeddings_model():
    """Loads the Hugging Face embeddings model."""
    with st.spinner("Downloading embeddings model... This may take a moment."):
        return download_hugging_face_embeddings()

llm = load_llm()
embeddings = load_embeddings_model()

# --- PROMPT TEMPLATE ---
system_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Please upload a medical PDF report to begin."}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- UI COMPONENTS ---
st.title("Medical Report Analysis Chatbot 💬")
st.markdown("Upload a medical PDF report on the left, then ask questions about it below.")

# --- SIDEBAR FOR FILE UPLOAD ---
with st.sidebar:
    st.header("Upload Report")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing your PDF report..."):
            try:
                # Read the PDF file from the uploaded bytes
                pdf_stream = io.BytesIO(uploaded_file.getvalue())
                pdf_reader = PdfReader(pdf_stream)
                
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

                if not text.strip():
                    st.error("Could not extract text from the PDF. The file might be empty or image-based.")
                else:
                    # Split the text into manageable chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_text(text)
                    
                    # Create the FAISS vector store from the text chunks
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    
                    # Save the vector store in the session state
                    st.session_state.vector_store = vector_store
                    
                    st.success("PDF processed successfully! You can now ask questions about the report.")
                    # Clear previous chat history on new upload
                    st.session_state.messages = [{"role": "assistant", "content": f"I've analyzed the report '{uploaded_file.name}'. How can I help you?"}]

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

# --- CHAT INTERFACE ---
# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about your medical report..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if a document has been processed
    if st.session_state.vector_store is None:
        with st.chat_message("assistant"):
            st.warning("Please upload a medical report first before asking questions.")
    else:
        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create the RAG chain
                    retriever = st.session_state.vector_store.as_retriever()
                    question_answer_chain = create_stuff_documents_chain(llm, system_prompt_template)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    
                    # Invoke the chain to get a response
                    response = rag_chain.invoke({"input": prompt})
                    answer = response.get("answer", "Sorry, I could not find an answer.")
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_message = f"An error occurred while generating a response: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
