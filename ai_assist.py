# -*- coding: utf-8 -*-
"""
Lt Cdr Shubham Mehta
AI Assist - v3 dated 19-12-2024
"""

import os
import subprocess
import logging
import pickle
from time import time
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document
from tqdm import tqdm
import faiss

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
SAVE_FILE = "doc_store.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"

# ASCII Art
MENU_BANNER = r"""
________                                       ______                       
___  __ \_____________________________________ ___  /                       
__  /_/ /  _ \_  ___/_  ___/  __ \_  __ \  __ `/_  /                        
_  ____//  __/  /   _(__  )/ /_/ /  / / / /_/ /_  /                         
/_/     \___//_/    /____/ \____//_/ /_/\__,_/ /_/                          
                                                                            
_______________   _______              _____       _____              _____ 
___    |___  _/   ___    |________________(_)________  /______ _________  /_
__  /| |__  /     __  /| |_  ___/_  ___/_  /__  ___/  __/  __ `/_  __ \  __/
_  ___ |_/ /      _  ___ |(__  )_(__  )_  / _(__  )/ /_ / /_/ /_  / / / /_  
/_/  |_/___/      /_/  |_/____/ /____/ /_/  /____/ \__/ \__,_/ /_/ /_/\__/  
                                                                            
"""

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index and Document Store
index = None
doc_store = {}
trained_files = set()

# Initialize FAISS
def initialize_faiss():
    global index
    if index is None:
        d = embedding_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(d)
        logging.info("FAISS index initialized.")

# Save knowledge
def save_knowledge():
    try:
        with open(SAVE_FILE, "wb") as f:
            pickle.dump({"doc_store": doc_store, "trained_files": list(trained_files)}, f)
        faiss.write_index(index, FAISS_INDEX_FILE)
        logging.info("Knowledge saved successfully.")
    except Exception as e:
        logging.error(f"Error saving knowledge: {e}")

# Flush knowledge
def flush_knowledge():
    global index, doc_store, trained_files
    initialize_faiss()
    doc_store = {}
    trained_files = set()
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)
    logging.info("All knowledge has been flushed.")

# Load knowledge
def load_knowledge():
    global index, doc_store, trained_files
    if os.path.exists(SAVE_FILE) and os.path.exists(FAISS_INDEX_FILE):
        try:
            with open(SAVE_FILE, "rb") as f:
                data = pickle.load(f)
                doc_store = data.get("doc_store", {})
                trained_files = set(data.get("trained_files", []))
            index = faiss.read_index(FAISS_INDEX_FILE)
            logging.info("Knowledge loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading knowledge: {e}")
            initialize_faiss()
    else:
        logging.warning("No saved knowledge found. Starting fresh.")
        initialize_faiss()

# Process documents
def split_into_chunks(text, max_chunk_size=300, overlap=50):
    words = text.split()
    chunks = [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size - overlap)]
    return chunks

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def add_to_index(file_name, chunks):
    global index, doc_store
    for idx, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk)
        index.add(embedding.reshape(1, -1))
        doc_store[len(doc_store)] = {"file_name": file_name, "chunk": chunk}
    trained_files.add(file_name)

def process_file(file_path):
    try:
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
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

# Search FAISS
def search_faiss(query, top_k=3):
    global index
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append(doc_store[idx]["chunk"])
    return " ".join(results)

# Query llama.cpp
def query_llama_cpp(prompt, n_predict=200):
    try:
        cmd = [
            r"D:\Projects\Chatbot\llama.cpp\bin\Release\llama-cli.exe",
            "--model",
            r"D:\Projects\Chatbot\llama.cpp\models\gemma-1.1-2b-it.Q3_K_M.gguf",
            "--prompt",
            prompt,
            "--n-predict",
            str(n_predict),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Error querying llama.cpp: {e}")
        return "An error occurred while generating the response."

# Answer a question
def answer_question(query):
    context = search_faiss(query, top_k=3)
    if not context:
        return "No relevant context found for your query."
    full_prompt = f"Question: {query}\nAnswer:"
    start_time = time()
    response = query_llama_cpp(full_prompt)
    elapsed_time = time() - start_time
    response_length = len(response.split())
    print(f"\nPerformance Metrics:")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Response length: {response_length} tokens")
    print(f"Query-processing rate: {response_length / elapsed_time:.2f} tokens/second\n")
    return response

# List trained documents
def list_trained_documents():
    if trained_files:
        print("\nTrained Documents:")
        for file in trained_files:
            print(f"- {file}")
    else:
        print("No documents have been trained yet.")

def interactive_menu():
    load_knowledge()
    while True:
        print(MENU_BANNER)
        print("1. Train\n2. List\n3. Query\n4. Remember\n5. Forget\n6. Exit")
        choice = input("Enter your choice (1/2/3/4/5/6): ").strip()

        if choice == "1":
            folder = input("Enter the folder path: ").strip()
            if os.path.isdir(folder):
                for root, _, files in os.walk(folder):
                    for file in tqdm(files, desc="Training files", unit="file"):
                        process_file(os.path.join(root, file))
                print("Training completed.")
            else:
                print("Invalid folder path.")
        elif choice == "2":
            list_trained_documents()
        elif choice == "3":
            query = input("Enter your query: ").strip()
            print("\nProcessing query...")
            answer = answer_question(query)
            print(f"{answer}")
        elif choice == "4":
            save_knowledge()
        elif choice == "5":
            flush_knowledge()
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    interactive_menu()
