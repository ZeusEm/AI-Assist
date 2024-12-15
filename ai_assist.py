# -*- coding: utf-8 -*-
"""
Lt Cdr Shubham Mehta
AI Assist - v0.1 dated 14-12-2024
"""

import os
import sqlite3
import pickle
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
from scipy.spatial.distance import cosine

# Set pytesseract path (modify this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\Shubham Mehta\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

# Initialize models
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Database setup
def init_db():
    conn = sqlite3.connect('ai_assist.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    file_name TEXT,
                    chunk_id INTEGER,
                    content TEXT,
                    embedding BLOB
                )''')
    conn.commit()
    conn.close()

# Split text into chunks
def split_into_chunks(text, max_chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = ' '.join(words[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks

# Store text and embeddings in database
def store_text(file_name, text):
    chunks = split_into_chunks(text)
    for idx, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk)
        conn = sqlite3.connect('ai_assist.db')
        c = conn.cursor()
        c.execute('INSERT INTO documents (file_name, chunk_id, content, embedding) VALUES (?, ?, ?, ?)',
                  (file_name, idx, chunk, pickle.dumps(embedding)))
        conn.commit()
        conn.close()

# Search for the most relevant text based on a query
def search_query(query, top_n=3):
    query_embedding = embedding_model.encode(query)
    conn = sqlite3.connect('ai_assist.db')
    c = conn.cursor()
    c.execute('SELECT content, embedding FROM documents')
    rows = c.fetchall()
    
    results = []
    for content, embedding_blob in rows:
        embeddings = pickle.loads(embedding_blob)
        similarity = 1 - cosine(embeddings, query_embedding)
        results.append((similarity, content))
    
    conn.close()

    # Sort results by similarity score in descending order
    results.sort(reverse=True, key=lambda x: x[0])
    
    # Aggregate top N results
    top_contexts = [res[1] for res in results[:top_n]]
    return ' '.join(top_contexts)

# Answer a question based on context
def answer_question(context, question):
    if not context.strip():
        return "I couldn't find relevant information."
    try:
        response = qa_pipeline(question=question, context=context)
        return response['answer']
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        text = ''
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''
        return text
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return ''

# Extract text from DOCX
def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error processing DOCX {file_path}: {e}")
        return ''

# Extract text from images (optional)
def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return ''

# Process a single file
def process_file(file_path):
    file_name = os.path.basename(file_path)
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif file_path.endswith((".jpg", ".png")):
        text = extract_text_from_image(file_path)
    else:
        print(f"Unsupported file format for {file_path}! Skipping.")
        return

    print(f"Processed file: {file_path}")
    store_text(file_name, text)

# Process a folder of files
def process_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            process_file(os.path.join(root, file))

# Interactive menu
def interactive_menu():
    while True:
        print("Welcome to the Personal AI Assistant!")
        print("1. Train the system with documents")
        print("2. Query the system")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == "1":
            folder_path = input("Enter the folder path: ").strip()
            if os.path.isdir(folder_path):
                process_folder(folder_path)
                print("Training completed.")
            else:
                print("Invalid folder path.")
        elif choice == "2":
            query = input("Enter your query: ").strip()
            context = search_query(query)
            answer = answer_question(context, query)
            print(f"Answer: {answer}")
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")

# Main function
def main():
    init_db()
    interactive_menu()

if __name__ == "__main__":
    main()
