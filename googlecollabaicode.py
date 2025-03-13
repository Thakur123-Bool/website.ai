import os
import gradio as gr
from PyPDF2 import PdfReader
from huggingface_hub import login
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
import re
import pandas as pd
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Hugging Face login
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
login(HUGGINGFACE_TOKEN)

# Load environment variables
load_dotenv()

# Set up the Llama pipeline
llama_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B",
    max_new_tokens=50,
    temperature=0.3,
    top_k=10,
    pad_token_id=0
)

# Strict prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""{question}\n\nContext:\n{context}\n\nAnswer:"""
)

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        logging.info(f"Processing file: {file_path}, Size: {file_size:.2f} MB")
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text, None
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return None, f"Error reading PDF: {e}"

# Detect and parse tabular data or enumerated lists
def detect_and_extract_tables_or_lists(text):
    try:
        lines = text.split("\n")
        table_data = []
        for line in lines:
            if re.match(r"^\S+(\s+\S+)*\s+\d+", line):
                table_data.append(re.split(r"\s{2,}", line.strip()))
        if table_data:
            return pd.DataFrame(table_data)
    except Exception as e:
        logging.error(f"Error extracting tables or lists: {e}")
    return None

# Split text into chunks
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# Create FAISS vectorstore
def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

# Initialize RetrievalQA
def initialize_retrieval_qa(vectorstore):
    llm = HuggingFacePipeline(pipeline=llama_pipeline)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}),
        chain_type_kwargs={"prompt": custom_prompt}
    )

# Clean response and format output
def format_response(question, response):
    try:
        response = re.sub(r"(Context:.*|Answer:)", "", response, flags=re.IGNORECASE).strip()
        if "compare" in question.lower() or "tabular form" in question.lower() or "table" in question.lower():
            tables_or_lists = detect_and_extract_tables_or_lists(response)
            if tables_or_lists is not None:
                return tables_or_lists.to_html(index=False, header=False)
        return f"<p>{response}</p>"
    except Exception as e:
        logging.error(f"Error formatting response: {e}")
        return response

# Process documents (files or URL)
def process_documents(uploaded_files=None, url=None):
    global retrieval_qa_chain
    text = ""
    status = []

    if uploaded_files and not url:  # Local file upload
        if not uploaded_files:
            return "Error: No files uploaded."
        status.append("Processing uploaded files...")
        for file in uploaded_files:
            file_text, error = extract_text_from_pdf(file.name)
            if error:
                return f"{error}\nStatus: {status[-1]}"
            text += file_text
            status.append(f"Processed file: {file.name}")
    elif url and not uploaded_files:  # URL download
        status.append(f"Downloading from URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return f"Error: Failed to download (Status {response.status_code})\nStatus: {status[-1]}"
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            file_text, error = extract_text_from_pdf("temp.pdf")
            if error:
                return f"{error}\nStatus: {status[-1]}"
            text = file_text
            os.remove("temp.pdf")
            status.append("Downloaded and processed PDF from URL")
        except Exception as e:
            return f"Error downloading from URL: {e}\nStatus: {status[-1]}"
    else:
        return "Error: Provide either files or a URL, not both."

    if not text.strip():
        return "Error: No text found in the documents.\nStatus: " + "\n".join(status)

    try:
        status.append("Splitting text into chunks...")
        text_chunks = split_text_into_chunks(text)
        status.append("Creating vectorstore...")
        vectorstore = create_vectorstore(text_chunks)
        status.append("Initializing QA chain...")
        retrieval_qa_chain = initialize_retrieval_qa(vectorstore)
        return "Documents processed successfully. Ask your questions!\nStatus: " + "\n".join(status)
    except Exception as e:
        logging.error(f"Error processing documents: {e}")
        return f"Error: {e}\nStatus: " + "\n".join(status)

# Answer questions
def answer_question(question, history):
    if 'retrieval_qa_chain' not in globals():
        return "Error: Documents have not been processed yet.", history
    try:
        response = retrieval_qa_chain.run({"query": question})
        formatted_response = format_response(question, response)
        history.append((question, formatted_response))
        return formatted_response, history
    except Exception as e:
        logging.error(f"Error answering question: {e}")
        return f"Error: {e}", history

# Update history display function
def update_history_display(history):
    history_html = "<h3>Conversation History</h3>"
    for i, (q, a) in enumerate(history):
        history_html += f"<p><b>Q{i+1}:</b> {q}</p><p><b>A{i+1}:</b> {a}</p><hr>"
    return history_html

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Chat with Your PDFs")

    with gr.Tab("Upload PDFs or URL"):
        with gr.Row():
            pdf_upload = gr.File(
                label="Upload PDF files (local use only, max 10 MB each)",
                file_types=[".pdf"],
                file_count="multiple"
            )
            url_input = gr.Textbox(
                label="Or enter a PDF URL (recommended for large files)",
                placeholder="https://example.com/sample.pdf"
            )
        process_button = gr.Button("Process Documents")
        process_output = gr.Textbox(label="Processing Status", interactive=False)

    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Ask a Question")
        answer_output = gr.HTML(label="Answer")
        chat_history = gr.HTML(label="Conversation History", value="<h3>Conversation History</h3>")

    # Shared history state
    history_state = gr.State([])

    # Link Gradio components
    process_button.click(
        process_documents,
        inputs=[pdf_upload, url_input],
        outputs=process_output
    )
    question_input.submit(
        lambda q, h: answer_question(q, h),
        inputs=[question_input, history_state],
        outputs=[answer_output, history_state]
    )
    question_input.submit(
        lambda h: update_history_display(h),
        inputs=history_state,
        outputs=chat_history
    )

# Launch app
app.launch(max_file_size="5mb")
