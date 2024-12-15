# -*- coding: utf-8 -*-
"""
Lt Cdr Shubham Mehta
AI Assist - v0.1 dated 14-12-2024
"""

import argparse
import sqlite3
import pickle
# pip install transformers
from transformers import pipeline
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image

# Initialize models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Database setup
def init_db():
    conn = sqlite3.connect('ai_assistant.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
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

# Extract text from Word document
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

# Extract text via OCR
def extract_text_via_ocr(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

# Store text and embeddings in database
def store_text(text):
    embeddings = embedding_model.encode([text])
    conn = sqlite3.connect('ai_assistant.db')
    c = conn.cursor()
    c.execute('INSERT INTO documents (content, embedding) VALUES (?, ?)', (text, pickle.dumps(embeddings)))
    conn.commit()
    conn.close()

# Search for the most relevant text based on a query
def search_query(query):
    query_embedding = embedding_model.encode([query])
    conn = sqlite3.connect('ai_assistant.db')
    c = conn.cursor()
    c.execute('SELECT content, embedding FROM documents')
    rows = c.fetchall()
    
    best_match = None
    best_similarity = -1

    for content, embedding_blob in rows:
        embeddings = pickle.loads(embedding_blob)
        similarity = (embeddings @ query_embedding.T).flatten()[0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = content

    conn.close()
    return best_match

# Answer a question based on context
def answer_question(context, question):
    response = qa_pipeline({'context': context, 'question': question})
    return response['answer']

# Main function
def main():
    parser = argparse.ArgumentParser(description="Personal AI Assistant")
    parser.add_argument("--file", help="Path to the document to process")
    parser.add_argument("--query", help="Query in plain English")
    parser.add_argument("--ocr", help="Path to an image for OCR", default=None)
    args = parser.parse_args()

    init_db()

    if args.file:
        if args.file.endswith(".pdf"):
            text = extract_text_from_pdf(args.file)
        elif args.file.endswith(".docx"):
            text = extract_text_from_docx(args.file)
        else:
            print("Unsupported file format! Only .pdf and .docx are supported.")
            return

        print("Extracted text:\n", text[:500], "...\n")  # Display a snippet of the extracted text
        store_text(text)
        print("Document processed and stored successfully.")

    if args.ocr:
        text = extract_text_via_ocr(args.ocr)
        print("Extracted text from image:\n", text[:500], "...\n")
        store_text(text)
        print("OCR result processed and stored successfully.")

    if args.query:
        context = search_query(args.query)
        if context:
            answer = answer_question(context, args.query)
            print(f"Answer: {answer}")
        else:
            print("No relevant context found for your query.")

if __name__ == "__main__":
    main()
