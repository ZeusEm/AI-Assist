# -*- coding: utf-8 -*-
"""
Lt Cdr Shubham Mehta
AI Assist - v0.1 dated 14-12-2024
"""

import os
import subprocess
import logging
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document
from tqdm import tqdm
import pickle
import faiss

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index and Document Store
index = None
doc_store = {}
trained_files = set()

# Paths for llama.cpp and model
LLAMA_CLI_PATH = r"D:\Projects\Chatbot\llama.cpp\bin\Release\llama-cli.exe"
MODEL_PATH = r"D:\Projects\Chatbot\llama.cpp\models\gemma-1.1-2b-it.Q3_K_M.gguf"

# Initialize FAISS
def initialize_faiss():
    global index
    d = embedding_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(d)

# Add documents to FAISS
def add_to_index(file_name, chunks):
    global doc_store
    for idx, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk)
        index.add(embedding.reshape(1, -1))
        doc_store[len(doc_store)] = {"file_name": file_name, "chunk": chunk}
    trained_files.add(file_name)

# Split text into chunks
def split_into_chunks(text, max_chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size - overlap):
        chunks.append(" ".join(words[i:i + max_chunk_size]))
    return chunks

# Extract text from PDF
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Extract text from DOCX
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Process file
def process_file(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        logging.warning(f"Unsupported file format: {file_path}")
        return
    chunks = split_into_chunks(text)
    add_to_index(os.path.basename(file_path), chunks)

# Search FAISS for relevant context
def search_faiss(query, top_k=3):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append(doc_store[idx]["chunk"])
    return " ".join(results)

# Query llama.cpp
def query_llama_cpp(prompt, n_predict=50):
    try:
        cmd = [
            LLAMA_CLI_PATH,
            "--model", MODEL_PATH,
            "--prompt", prompt,
            "--n-predict", str(n_predict)
        ]
        # Run the llama-cli executable and capture the output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        return output
    except subprocess.CalledProcessError as e:
        logging.error(f"Error querying llama.cpp: {e}")
        return "An error occurred while generating the response."

# Answer a question
def answer_question(query):
    context = search_faiss(query)
    if not context:
        return "No relevant context found for your query."
    logging.info("Context retrieved successfully.")
    full_prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = query_llama_cpp(full_prompt, n_predict=100)
    return response

# List trained documents
def list_trained_documents():
    if trained_files:
        print("\nTrained Documents:")
        for file in trained_files:
            print(f"- {file}")
    else:
        print("No documents have been trained yet.")

# Interactive menu
def interactive_menu():
    initialize_faiss()
    while True:
        print("\nWelcome to the Personal AI Assistant!")
        print("1. Train the system with documents")
        print("2. Query the system")
        print("3. List trained documents")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == "1":
            folder = input("Enter the folder path: ").strip()
            if os.path.isdir(folder):
                print("Processing documents...")
                for root, _, files in os.walk(folder):
                    for file in tqdm(files, desc="Training files", unit="file"):
                        process_file(os.path.join(root, file))
                print("Training completed.")
            else:
                print("Invalid folder path.")
        elif choice == "2":
            query = input("Enter your query: ").strip()
            print("Processing query...")
            answer = answer_question(query)
            print(f"Answer:\n{answer}")
        elif choice == "3":
            list_trained_documents()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")

# Main function
if __name__ == "__main__":
    interactive_menu()
