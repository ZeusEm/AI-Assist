# -*- coding: utf-8 -*-
"""
Lt Cdr Shubham Mehta
AI Assist - v0.1 dated 14-12-2024
"""

import os
import pickle
import logging
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document
from tqdm import tqdm

# Initialize logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Models for QA and Summarization
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# FAISS Index and Document Store
index = None
doc_store = {}
trained_files = set()  # Store file names for listing

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

# Preprocessing: Chunk text with overlap
def split_into_chunks(text, max_chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size - overlap):
        chunks.append(" ".join(words[i:i + max_chunk_size]))
    return chunks

# Process files
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

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

# Search in FAISS
def search_faiss(query, top_k=5):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append(doc_store[idx]["chunk"])
    return results

# Summarize retrieved context
def summarize_context(contexts, max_length=512):
    summarized_chunks = []
    for i in range(0, len(contexts), 3):  # Process in batches of 3 chunks
        batch = " ".join(contexts[i:i + 3])
        try:
            summary = summarizer_pipeline(batch, max_length=max_length, truncation=True)
            summarized_chunks.append(summary[0]["summary_text"])
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
    return " ".join(summarized_chunks)

# Answer questions with enhanced QA
def answer_question(query):
    results = search_faiss(query, top_k=10)  # Retrieve more chunks
    if not results:
        return "No relevant context found for your query."
    
    # Rank contexts by similarity and aggregate top ones
    ranked_contexts = [res for res in results]
    
    # Summarize retrieved contexts
    try:
        summarized_context = summarize_context(ranked_contexts[:5])  # Use top 5 chunks for summarization
        logging.info(f"Summarized context: {summarized_context}")
        answer = qa_pipeline(f"Question: {query} Context: {summarized_context}", max_length=512, truncation=True)
        return f"{answer[0]['generated_text']}\n\n[Summary of Context]\n{summarized_context}"
    except Exception as e:
        logging.error(f"QA or Summarization failed: {e}")
        return "An error occurred while generating the answer."

# View trained documents
def list_trained_documents():
    if trained_files:
        print("Trained Documents:")
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
