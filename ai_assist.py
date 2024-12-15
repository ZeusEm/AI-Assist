# -*- coding: utf-8 -*-
"""
Lt Cdr Shubham Mehta
AI Assist - v0.1 dated 14-12-2024
"""

import argparse
import os
import sqlite3
import pickle
import subprocess
import sys
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image

# Helper function to install packages
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install required packages
install_and_import("transformers")
install_and_import("sentence-transformers")
install_and_import("pdfplumber")
install_and_import("python-docx")
install_and_import("pytesseract")
install_and_import("Pillow")

# Function to check if Tesseract is installed
def check_tesseract_installed():
    tesseract_path = r'C:\Users\Shubham Mehta\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    if not os.path.exists(tesseract_path):
        print("Tesseract is not installed at the specified path.")
        print("Please download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("Once installed, ensure it is located at C:\\Users\\Shubham Mehta\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe")
        sys.exit(1)
    return tesseract_path

# Set the path for Tesseract executable
tesseract_cmd_path = check_tesseract_installed()
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

# Clear terminal
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# Initialize models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Database setup
def init_db():
    conn = sqlite3.connect('ai_assist.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    file_name TEXT,
                    content TEXT,
                    embedding BLOB
                )''')
    conn.commit()
    conn.close()

# Extract text from PDF
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return text

# Check if PDF is OCR-readable
def is_pdf_readable(file_path):
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                return True
    return False

# Perform OCR on PDF
def extract_text_from_pdf_via_ocr(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            image = page.to_image()
            text += pytesseract.image_to_string(image.original)
    return text

# Extract text from Word document
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

# Extract text via OCR from images
def extract_text_via_ocr(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

# Store text and embeddings in database
def store_text(file_name, text):
    embeddings = embedding_model.encode([text])
    conn = sqlite3.connect('ai_assist.db')
    c = conn.cursor()
    c.execute('INSERT INTO documents (file_name, content, embedding) VALUES (?, ?, ?)', (file_name, text, pickle.dumps(embeddings)))
    conn.commit()
    conn.close()

# Search for the most relevant text based on a query
def search_query(query):
    query_embedding = embedding_model.encode([query])
    conn = sqlite3.connect('ai_assist.db')
    c = conn.cursor()
    c.execute('SELECT file_name, content, embedding FROM documents')
    rows = c.fetchall()
    
    best_matches = []

    for file_name, content, embedding_blob in rows:
        embeddings = pickle.loads(embedding_blob)
        similarity = (embeddings @ query_embedding.T).flatten()[0]
        best_matches.append((similarity, file_name, content))

    best_matches.sort(reverse=True, key=lambda x: x[0])
    conn.close()
    return best_matches

# Generate a response based on the best matches and input query
def generate_response(query):
    best_matches = search_query(query)
    if best_matches:
        response_text = ""
        for _, file_name, context in best_matches[:3]:  # Take top 3 matches
            answer = answer_question(context, query)
            response_text += f"Answer from {file_name}:\n{answer}\n\n"
        return response_text
    else:
        return "No relevant context found for your query."

# Process a single file
def process_file(file_path):
    file_name = os.path.basename(file_path)
    if file_path.lower().endswith((".pdf", ".docx", ".png", ".jpg", ".jpeg")):
        if file_path.endswith(".pdf"):
            if is_pdf_readable(file_path):
                text = extract_text_from_pdf(file_path)
            else:
                text = extract_text_from_pdf_via_ocr(file_path)
        elif file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_via_ocr(file_path)
        
        print(f"Processed file: {file_path}")
        store_text(file_name, text)
    else:
        print(f"Unsupported file format for {file_path}! Skipping.")

# Process a folder of files
def process_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            process_file(os.path.join(root, file))

# List all trained documents
def list_trained_documents():
    conn = sqlite3.connect('ai_assist.db')
    c = conn.cursor()
    c.execute('SELECT file_name FROM documents')
    rows = c.fetchall()
    conn.close()
    if rows:
        print("Trained Documents:")
        for row in rows:
            print(f"- {row[0]}")
    else:
        print("No documents have been trained yet.")

# Interactive menu
def interactive_menu():
    while True:
        clear_terminal()
        print("Welcome to the Personal AI Assistant!")
        print("Select an option:")
        print("1. Train the system with documents")
        print("2. Query the system")
        print("3. List trained documents")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ").strip()
        
        if choice == "1":
            folder_path = input("Enter the path to the folder containing documents: ").strip()
            if os.path.isdir(folder_path):
                init_db()  # Ensure the database is initialized
                process_folder(folder_path)
                print("Training completed.")
            else:
                print("Invalid folder path.")
            input("Press Enter to continue...")
        elif choice == "2":
            query = input("Enter your query: ").strip()
            response = generate_response(query)
            print("\nGenerated Response:")
            print(response)
            input("Press Enter to continue...")
        elif choice == "3":
            list_trained_documents()
            input("Press Enter to continue...")
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")

# Main function
def main():
    init_db()
    interactive_menu()

if __name__ == "__main__":
    main()
