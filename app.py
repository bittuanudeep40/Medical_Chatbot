from flask import Flask, render_template, jsonify, request, session
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
import tempfile
import uuid

app = Flask(__name__)

# It's important to set a strong, secret key in a real application
app.secret_key = 'your_super_secret_key_change_me'

load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("Downloading Hugging Face embeddings...")
embeddings = download_hugging_face_embeddings()
print("Embeddings downloaded successfully.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4, max_output_tokens=500)
system_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# --- Routes ---

@app.route("/")
def index():
    session.clear()
    return render_template('chat.html')

# MODIFIED: Route to handle PDF file uploads more robustly
@app.route("/upload", methods=["POST"])
def upload_file():
    print("\n--- Received a file upload request ---")
    if 'pdfFile' not in request.files:
        print("Error: 'pdfFile' not in request.files")
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['pdfFile']
    if file.filename == '':
        print("Error: No file selected.")
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        try:
            print(f"Processing file: {file.filename}")
            pdf_stream = io.BytesIO(file.read())
            pdf_reader = PdfReader(pdf_stream)
            
            text = ""
            print("Extracting text from PDF...")
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            print(f"Extracted {len(text)} characters.")
            
            if not text.strip():
                 print("Error: No text could be extracted from the PDF.")
                 return jsonify({"status": "error", "message": "Could not extract text from PDF. The file might be empty or scanned."}), 400

            # --- NEW LOGIC: Process the document and create the vector store here ---
            print("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            print(f"Created {len(chunks)} chunks.")

            print("Creating FAISS vector store from chunks. This may take a moment...")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            print("Vector store created successfully.")

            # Save the vector store to a temporary directory
            temp_dir = tempfile.gettempdir()
            unique_folder_name = str(uuid.uuid4())
            db_path = os.path.join(temp_dir, unique_folder_name)
            os.makedirs(db_path, exist_ok=True) # Create the directory

            print(f"Saving vector store to: {db_path}")
            vector_store.save_local(db_path)
            print("Vector store saved.")

            # Store the path to the FAISS index in the session
            session['db_path'] = db_path
            
            print("--- File upload and processing successful ---")
            return jsonify({"status": "success"})
        except Exception as e:
            print(f"!!! An exception occurred during upload/processing: {e} !!!")
            return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500
            
    return jsonify({"status": "error", "message": "Invalid file type. Please upload a PDF."}), 400

# MODIFIED: Chat route now loads the pre-processed vector store
@app.route("/get", methods=["POST"])
def chat():
    print("\n--- Received a chat message ---")
    user_input = request.form.get("msg")
    
    db_path = session.get('db_path')
    print(f"Retrieved DB path from session: {db_path}")

    if not user_input:
        print("Error: No message received.")
        return "Error: No message received."
    
    if not db_path or not os.path.exists(db_path):
        print("Error: DB path not found or invalid.")
        return "Error: No report has been uploaded or the session has expired. Please refresh and upload a new file."

    try:
        # Load the FAISS vector store from disk
        print("Loading FAISS vector store from disk...")
        # The allow_dangerous_deserialization flag is needed for loading FAISS indexes.
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")
        
        retriever = vector_store.as_retriever()
        
        print("Creating RAG chain...")
        question_answer_chain = create_stuff_documents_chain(llm, system_prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        print(f"Invoking RAG chain with input: '{user_input}'")
        response = rag_chain.invoke({"input": user_input})
        print("RAG chain invocation complete.")
        
        answer = response.get("answer", "Sorry, I could not find an answer.")
        return str(answer)
    
    except Exception as e:
        print(f"!!! An exception occurred during chat processing: {e} !!!")
        return "An error occurred while processing your request. Please try again."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

