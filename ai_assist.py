# -*- coding: utf-8 -*-
"""
Lt Cdr Shubham Mehta
AI Assist - v4 dated 20-12-2024
"""

# Import necessary libraries

import os  # Provides a way of using operating system-dependent functionality, such as reading or writing to the file system

import subprocess  # Allows spawning new processes, connecting to their input/output/error pipes, and obtaining their return codes

import logging  # Provides a flexible framework for emitting log messages from Python programs

import pickle  # Implements binary protocols for serializing and de-serializing a Python object structure

import hashlib  # Provides a common interface to many secure hash and message digest algorithms, including SHA256

from time import time  # Provides various time-related functions, including measuring performance metrics

from sentence_transformers import SentenceTransformer  # Library for creating sentence and text embeddings, ideal for NLP tasks

import pdfplumber  # Library for reading the text content of a PDF file

from docx import Document  # Library for working with Microsoft Word (.docx) files, enabling text extraction and document manipulation

from tqdm import tqdm  # Library for creating progress bars to visually indicate progress in loops and tasks

import faiss  # Library for efficient similarity search and clustering of dense vectors, developed by Facebook AI Research

from PIL import Image  # The Python Imaging Library (PIL) forked as Pillow, provides image processing capabilities

import pytesseract  # Python wrapper for Google's Tesseract-OCR Engine, used for optical character recognition (OCR) to extract text from images


# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Purpose: This line initializes the logging module to track events and errors that occur during the execution of the program.
# - level=logging.INFO: Sets the logging level to INFO, which means that all messages with severity level INFO and above (INFO, WARNING, ERROR, CRITICAL) will be logged.
# - format="%(asctime)s - %(levelname)s - %(message)s": Specifies the format of the log messages, including the timestamp, the log level, and the actual log message. This helps in making log messages more informative and easier to debug.

# File paths
SAVE_FILE = "doc_store.pkl"  # Path to the file where the document store will be saved
FAISS_INDEX_FILE = "faiss_index.bin"  # Path to the file where the FAISS index will be saved
HASH_FILE = "file_hashes.pkl"  # Path to the file where file hashes will be saved

# Purpose: These lines define the file paths used to save and load the state of the application.
# - SAVE_FILE: The document store (a dictionary containing the text chunks and metadata) is saved to this file using the pickle module.
# - FAISS_INDEX_FILE: The FAISS index, which is used for efficient similarity search of dense vectors, is saved to this file.
# - HASH_FILE: The hashes of the processed files are saved to this file to detect changes in the files and avoid unnecessary reprocessing.


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

# Purpose: This line initializes the SentenceTransformer model for creating sentence and text embeddings.
# - SentenceTransformer("all-MiniLM-L6-v2"): Loads the pre-trained "all-MiniLM-L6-v2" model, which is specifically designed for transforming sentences into embeddings.
# - Embeddings are essential for representing text in a numerical format that can be used for various NLP tasks, such as similarity search and clustering.

# FAISS Index and Document Store
index = None  # Initializes the FAISS index to None, to be set up later
doc_store = {}  # Initializes an empty dictionary to store document chunks and their metadata
trained_files = set()  # Initializes an empty set to keep track of files that have been processed and trained
file_hashes = {}  # Initializes an empty dictionary to store the hashes of processed files for change detection

# Purpose: These lines initialize global variables that will be used throughout the script.
# - index: A FAISS index object that will be used for efficient similarity search of dense vectors.
# - doc_store: A dictionary that will store text chunks and their metadata, serving as the document store.
# - trained_files: A set that will keep track of files that have already been processed and trained to avoid reprocessing.
# - file_hashes: A dictionary that will store the hashes of processed files to detect changes and process only new or modified files.


# Initialize FAISS
def initialize_faiss():
    global index
    if index is None:
        # Retrieve the embedding dimension from the SentenceTransformer model
        d = embedding_model.get_sentence_embedding_dimension()
        # Initialize a flat (non-hierarchical) FAISS index that uses L2 (Euclidean) distance
        index = faiss.IndexFlatL2(d)
        # Log that the FAISS index has been successfully initialized
        logging.info("FAISS index initialized.")

# Purpose: This function initializes the FAISS index, which is used for efficient similarity search of dense vectors.
# - The 'global index' statement ensures that we modify the global 'index' variable defined earlier.
# - The 'if index is None' check ensures that the index is only initialized once.
# - 'embedding_model.get_sentence_embedding_dimension()' retrieves the dimensionality of the embeddings created by the SentenceTransformer model.
# - 'faiss.IndexFlatL2(d)' initializes a flat FAISS index with L2 (Euclidean) distance, which is effective for finding the nearest neighbors in high-dimensional space.
# - 'logging.info("FAISS index initialized.")' logs the initialization event, helping in tracking the flow and debugging.


# Save knowledge
def save_knowledge():
    try:
        # Save the document store and trained files list to a pickle file
        with open(SAVE_FILE, "wb") as f:
            pickle.dump({"doc_store": doc_store, "trained_files": list(trained_files)}, f)
        
        # Save the FAISS index to a file
        faiss.write_index(index, FAISS_INDEX_FILE)
        
        # Save the file hashes to a pickle file
        with open(HASH_FILE, "wb") as f:
            pickle.dump(file_hashes, f)
        
        # Log a success message indicating that knowledge has been saved successfully
        logging.info("Knowledge saved successfully.")
    
    # Handle any exceptions that occur during the saving process
    except Exception as e:
        # Log an error message with the exception details
        logging.error(f"Error saving knowledge: {e}")

# Purpose: This function saves the current state of the document store, FAISS index, and file hashes to files for persistent storage.
# - The 'try' block attempts to save the data, and the 'except' block handles any errors that occur.
# - 'with open(SAVE_FILE, "wb") as f': Opens the SAVE_FILE in write-binary mode and assigns the file object to 'f'.
# - 'pickle.dump({"doc_store": doc_store, "trained_files": list(trained_files)}, f)': Serializes the 'doc_store' dictionary and 'trained_files' set to the file using the pickle module.
# - 'faiss.write_index(index, FAISS_INDEX_FILE)': Writes the FAISS index to the FAISS_INDEX_FILE, preserving the state of the index.
# - 'with open(HASH_FILE, "wb") as f': Opens the HASH_FILE in write-binary mode and assigns the file object to 'f'.
# - 'pickle.dump(file_hashes, f)': Serializes the 'file_hashes' dictionary to the file using the pickle module.
# - 'logging.info("Knowledge saved successfully.")': Logs a success message if the knowledge is saved without any issues.
# - 'except Exception as e': Catches any exceptions that occur during the save process.
# - 'logging.error(f"Error saving knowledge: {e}")': Logs an error message with the exception details if an error occurs.


# Save knowledge
def save_knowledge():
    try:
        # Save the document store and trained files list to a pickle file
        with open(SAVE_FILE, "wb") as f:
            pickle.dump({"doc_store": doc_store, "trained_files": list(trained_files)}, f)
        
        # Save the FAISS index to a file
        faiss.write_index(index, FAISS_INDEX_FILE)
        
        # Save the file hashes to a pickle file
        with open(HASH_FILE, "wb") as f:
            pickle.dump(file_hashes, f)
        
        # Log a success message indicating that knowledge has been saved successfully
        logging.info("Knowledge saved successfully.")
    
    # Handle any exceptions that occur during the saving process
    except Exception as e:
        # Log an error message with the exception details
        logging.error(f"Error saving knowledge: {e}")

# Purpose: This function saves the current state of the document store, FAISS index, and file hashes to files for persistent storage.
# - The 'try' block attempts to save the data, and the 'except' block handles any errors that occur.
# - 'with open(SAVE_FILE, "wb") as f': Opens the SAVE_FILE in write-binary mode and assigns the file object to 'f'.
# - 'pickle.dump({"doc_store": doc_store, "trained_files": list(trained_files)}, f)': Serializes the 'doc_store' dictionary and 'trained_files' set to the file using the pickle module.
# - 'faiss.write_index(index, FAISS_INDEX_FILE)': Writes the FAISS index to the FAISS_INDEX_FILE, preserving the state of the index.
# - 'with open(HASH_FILE, "wb") as f': Opens the HASH_FILE in write-binary mode and assigns the file object to 'f'.
# - 'pickle.dump(file_hashes, f)': Serializes the 'file_hashes' dictionary to the file using the pickle module.
# - 'logging.info("Knowledge saved successfully.")': Logs a success message if the knowledge is saved without any issues.
# - 'except Exception as e': Catches any exceptions that occur during the save process.
# - 'logging.error(f"Error saving knowledge: {e}")': Logs an error message with the exception details if an error occurs.


# Flush knowledge
def flush_knowledge():
    """Flush (clear) all stored knowledge, effectively resetting the document store, FAISS index, and file hashes.
    
    This function clears all stored data, re-initializes the FAISS index, and removes the saved files. It resets the global variables for the document store, trained files, and file hashes.
    """
    global index, doc_store, trained_files, file_hashes  # Use global variables for the FAISS index, document store, trained files, and file hashes

    # Re-initialize the FAISS index to ensure it is reset
    initialize_faiss()

    # Clear the document store by setting it to an empty dictionary
    doc_store = {}

    # Reset the set of trained files to an empty set
    trained_files = set()

    # Clear the file hashes by setting it to an empty dictionary
    file_hashes = {}

    # Check if the SAVE_FILE exists and remove it if it does
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)

    # Check if the FAISS_INDEX_FILE exists and remove it if it does
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)

    # Check if the HASH_FILE exists and remove it if it does
    if os.path.exists(HASH_FILE):
        os.remove(HASH_FILE)

    # Log a message indicating that all knowledge has been successfully flushed
    logging.info("All knowledge has been flushed.")

# Purpose: This function clears all stored knowledge, effectively resetting the document store, FAISS index, and file hashes.
# - global index, doc_store, trained_files, file_hashes: Ensures that the function uses the global variables for the FAISS index, document store, trained files, and file hashes.
# - initialize_faiss(): Re-initializes the FAISS index to ensure it is reset.
# - doc_store = {}: Clears the document store by setting it to an empty dictionary.
# - trained_files = set(): Resets the set of trained files to an empty set.
# - file_hashes = {}: Clears the file hashes by setting it to an empty dictionary.
# - if os.path.exists(SAVE_FILE): os.remove(SAVE_FILE): Checks if the SAVE_FILE exists and removes it if it does, ensuring no old data persists.
# - if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE): Checks if the FAISS_INDEX_FILE exists and removes it if it does, ensuring the index is reset.
# - if os.path.exists(HASH_FILE): os.remove(HASH_FILE): Checks if the HASH_FILE exists and removes it if it does, ensuring no old hash data persists.
# - logging.info("All knowledge has been flushed."): Logs a message indicating that all knowledge has been successfully flushed.


# Load knowledge
def load_knowledge():
    global index, doc_store, trained_files, file_hashes
    
    # Check if the save files for the document store, FAISS index, and file hashes exist
    if os.path.exists(SAVE_FILE) and os.path.exists(FAISS_INDEX_FILE) and os.path.exists(HASH_FILE):
        try:
            # Load the document store and trained files from the pickle file
            with open(SAVE_FILE, "rb") as f:
                data = pickle.load(f)
                doc_store = data.get("doc_store", {})
                trained_files = set(data.get("trained_files", []))
            
            # Load the file hashes from the pickle file
            with open(HASH_FILE, "rb") as f:
                file_hashes = pickle.load(f)
            
            # Load the FAISS index from the file
            index = faiss.read_index(FAISS_INDEX_FILE)
            
            # Log a success message indicating that knowledge has been loaded successfully
            logging.info("Knowledge loaded successfully.")
        
        # Handle any exceptions that occur during the loading process
        except Exception as e:
            # Log an error message with the exception details
            logging.error(f"Error loading knowledge: {e}")
            # Re-initialize the FAISS index if an error occurs
            initialize_faiss()
    
    # If the save files do not exist, initialize a fresh FAISS index
    else:
        logging.warning("No saved knowledge found. Starting fresh.")
        initialize_faiss()

# Purpose: This function loads the previously saved state of the document store, FAISS index, and file hashes from files, if they exist.
# - 'global index, doc_store, trained_files, file_hashes': Ensures that the function modifies the global variables defined earlier.
# - 'if os.path.exists(SAVE_FILE) and os.path.exists(FAISS_INDEX_FILE) and os.path.exists(HASH_FILE)': Checks if the necessary save files exist.
# - 'with open(SAVE_FILE, "rb") as f': Opens the SAVE_FILE in read-binary mode and assigns the file object to 'f'.
# - 'data = pickle.load(f)': Deserializes the data from the SAVE_FILE using the pickle module.
# - 'doc_store = data.get("doc_store", {})': Retrieves the document store from the loaded data, defaulting to an empty dictionary if not found.
# - 'trained_files = set(data.get("trained_files", []))': Retrieves the trained files from the loaded data, defaulting to an empty set if not found.
# - 'with open(HASH_FILE, "rb") as f': Opens the HASH_FILE in read-binary mode and assigns the file object to 'f'.
# - 'file_hashes = pickle.load(f)': Deserializes the file hashes from the HASH_FILE using the pickle module.
# - 'index = faiss.read_index(FAISS_INDEX_FILE)': Reads the FAISS index from the FAISS_INDEX_FILE.
# - 'logging.info("Knowledge loaded successfully.")': Logs a success message if the knowledge is loaded without any issues.
# - 'except Exception as e': Catches any exceptions that occur during the load process.
# - 'logging.error(f"Error loading knowledge: {e}")': Logs an error message with the exception details if an error occurs.
# - 'initialize_faiss()': Re-initializes the FAISS index if an error occurs during loading or if the save files do not exist.
# - 'logging.warning("No saved knowledge found. Starting fresh.")': Logs a warning message if no save files are found, indicating that a fresh initialization is done.


# Process documents

def split_into_chunks(text, max_chunk_size=300, overlap=50):
    """Split text into smaller chunks for embedding and indexing.
    
    Args:
        text (str): The input text to be split into chunks.
        max_chunk_size (int): The maximum size of each chunk in terms of words. Default is 300.
        overlap (int): The number of words to overlap between consecutive chunks. Default is 50.
        
    Returns:
        list of str: A list containing the text chunks.
    """
    words = text.split()  # Split the text into individual words
    # Create chunks of words with specified max size and overlap
    chunks = [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size - overlap)]
    return chunks

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        str: The extracted text from the PDF file.
    """
    with pdfplumber.open(file_path) as pdf:  # Open the PDF file
        # Extract text from all pages and join them into a single string
        return "".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(file_path):
    """Extract text from a Microsoft Word (.docx) file.
    
    Args:
        file_path (str): The path to the .docx file.
        
    Returns:
        str: The extracted text from the .docx file.
    """
    doc = Document(file_path)  # Open the .docx file
    # Extract text from all paragraphs and join them into a single string
    return "\n".join([para.text for para in doc.paragraphs])

# Purpose: These functions handle the processing of documents, splitting text into smaller chunks, and extracting text from PDF and Microsoft Word files.
# - split_into_chunks(text, max_chunk_size=300, overlap=50): Splits the input text into smaller chunks for embedding and indexing, using specified maximum chunk size and overlap.
# - extract_text_from_pdf(file_path): Extracts text from all pages of a PDF file and joins them into a single string using pdfplumber.
# - extract_text_from_docx(file_path): Extracts text from all paragraphs of a Microsoft Word (.docx) file and joins them into a single string using python-docx.

def extract_text_from_image(file_path):
    """Extract text from an image file using Optical Character Recognition (OCR).
    
    Args:
        file_path (str): The path to the image file.
        
    Returns:
        str: The extracted text from the image.
    """
    # Use pytesseract to perform OCR on the image and extract text
    text = pytesseract.image_to_string(Image.open(file_path))
    return text

def add_to_index(file_name, chunks):
    """Add text chunks to the FAISS index and document store.
    
    Args:
        file_name (str): The name of the file from which the chunks are extracted.
        chunks (list of str): A list of text chunks to be indexed.
    """
    global index, doc_store
    for idx, chunk in enumerate(chunks):
        # Generate embedding for the text chunk
        embedding = embedding_model.encode(chunk)
        # Add the embedding to the FAISS index
        index.add(embedding.reshape(1, -1))
        # Store the chunk with its metadata in the document store
        doc_store[len(doc_store)] = {"file_name": file_name, "chunk": chunk}
    # Add the file name to the set of trained files to keep track of processed files
    trained_files.add(file_name)

# Purpose: These functions handle extracting text from image files and adding text chunks to the FAISS index and document store.
# - extract_text_from_image(file_path): Uses pytesseract to perform OCR on the image and extract text, which is then returned as a string.
# - add_to_index(file_name, chunks): Adds text chunks to the FAISS index and document store. For each chunk, it generates an embedding using the SentenceTransformer model, adds the embedding to the FAISS index, and stores the chunk along with its metadata in the document store. The file name is added to the set of trained files to track processed files.

def process_file(file_path):
    """Process a single file and extract text based on the file type.
    
    Args:
        file_path (str): The path to the file to be processed.
    """
    try:
        text = ""  # Initialize an empty string for the extracted text

        # Check the file extension and extract text accordingly
        if file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)  # Extract text from a PDF file
        elif file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)  # Extract text from a Microsoft Word (.docx) file
        elif file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
            text = extract_text_from_image(file_path)  # Extract text from an image file using OCR
        else:
            logging.warning(f"Unsupported file format: {file_path}")  # Log a warning if the file format is unsupported
            return

        # Split the extracted text into chunks
        chunks = split_into_chunks(text)
        # Add the text chunks to the FAISS index and document store
        add_to_index(os.path.basename(file_path), chunks)

    except Exception as e:
        # Log an error message if an exception occurs
        logging.error(f"Error processing file {file_path}: {e}")

def compute_file_hash(file_path):
    """Compute a SHA-256 hash for a file to detect changes.
    
    Args:
        file_path (str): The path to the file for which the hash is to be computed.
        
    Returns:
        str: The computed SHA-256 hash of the file.
    """
    hasher = hashlib.sha256()  # Initialize a SHA-256 hash object
    with open(file_path, "rb") as f:
        buf = f.read()  # Read the contents of the file
        hasher.update(buf)  # Update the hash object with the file contents
    return hasher.hexdigest()  # Return the hexadecimal representation of the hash

# Purpose: These functions handle processing files based on their type and computing SHA-256 hashes for change detection.
# - process_file(file_path): Processes a single file, extracts text based on the file type (PDF, .docx, or image), splits the text into chunks, and adds the chunks to the FAISS index and document store. Logs an error message if any exception occurs.
# - compute_file_hash(file_path): Computes a SHA-256 hash for a file to detect changes. Reads the file in binary mode, updates the hash object with the file contents, and returns the hexadecimal representation of the hash.


def scan_and_process_files(folder_path):
    """Scan a folder and process new or modified files based on their hashes.
    
    Args:
        folder_path (str): The path to the folder containing the files to be scanned and processed.
    """
    global file_hashes  # Use the global 'file_hashes' dictionary to track file hashes

    # Walk through the directory structure starting from 'folder_path'
    for root, _, files in os.walk(folder_path):
        # Use tqdm to display a progress bar for scanning files
        for file in tqdm(files, desc="Scanning files", unit="file"):
            file_path = os.path.join(root, file)  # Get the full file path

            # Check if the file has a supported extension
            if file.lower().endswith((".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
                file_hash = compute_file_hash(file_path)  # Compute the hash of the file

                # Check if the file is already processed and unchanged
                if file_path in file_hashes and file_hashes[file_path] == file_hash:
                    logging.info(f"Skipping unchanged file: {file_path}")  # Log that the file is skipped
                else:
                    logging.info(f"Processing new/modified file: {file_path}")  # Log that the file is being processed
                    process_file(file_path)  # Process the file
                    file_hashes[file_path] = file_hash  # Update the hash of the file in the dictionary
            else:
                logging.info(f"Ignoring unsupported file: {file_path}")  # Log that the file format is unsupported

# Purpose: This function scans a specified folder for files, checks if they have been modified, and processes only the new or modified files.
# - global file_hashes: Ensures that the function uses the global 'file_hashes' dictionary to track file hashes.
# - os.walk(folder_path): Walks through the directory structure starting from 'folder_path'.
# - for file in tqdm(files, desc="Scanning files", unit="file"): Uses tqdm to display a progress bar for scanning files.
# - os.path.join(root, file): Gets the full file path by joining the root directory with the file name.
# - file.lower().endswith((".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")): Checks if the file has a supported extension.
# - compute_file_hash(file_path): Computes the SHA-256 hash of the file.
# - if file_path in file_hashes and file_hashes[file_path] == file_hash: Checks if the file is already processed and unchanged.
# - logging.info(f"Skipping unchanged file: {file_path}"): Logs that the file is skipped if it is unchanged.
# - logging.info(f"Processing new/modified file: {file_path}"): Logs that the file is being processed if it is new or modified.
# - process_file(file_path): Processes the file by extracting text and adding it to the index.
# - file_hashes[file_path] = file_hash: Updates the hash of the file in the 'file_hashes' dictionary to keep track of the processed state.
# - logging.info(f"Ignoring unsupported file: {file_path}"): Logs that the file format is unsupported if it does not match the specified extensions.


# Search FAISS
def search_faiss(query, top_k=3):
    """Search the FAISS index for relevant text based on a query.
    
    Args:
        query (str): The search query to find relevant text.
        top_k (int): The number of top results to return. Default is 3.
        
    Returns:
        str: The concatenated text chunks of the top search results.
    """
    global index  # Use the global 'index' variable for the FAISS index

    # Encode the query into an embedding
    query_embedding = embedding_model.encode(query).reshape(1, -1)

    # Perform the search on the FAISS index to find the top_k nearest neighbors
    distances, indices = index.search(query_embedding, top_k)

    results = []  # Initialize an empty list to store the search results

    # Iterate over the search results (distances and indices)
    for dist, idx in zip(distances[0], indices[0]):
        if idx in doc_store:
            # If the index exists in the document store, append the chunk to the results
            results.append(doc_store[idx]["chunk"])
        else:
            # If the index is not found in the document store, log a debug message and remove the ID from the FAISS index
            logging.debug(f"Index {idx} not found in document store. Removing from FAISS index.")
            try:
                index.remove_ids(faiss.IDSelectorRange(idx, idx + 1))
            except Exception as e:
                # Log a debug message if an error occurs while removing the ID from the FAISS index
                logging.debug(f"Error removing ID {idx} from FAISS index: {e}")

    # Return the concatenated text chunks of the top search results
    return " ".join(results)

# Purpose: This function searches the FAISS index for relevant text based on a query and returns the top search results.
# - global index: Ensures that the function uses the global 'index' variable for the FAISS index.
# - query_embedding = embedding_model.encode(query).reshape(1, -1): Encodes the query into an embedding and reshapes it to be a 2D array with one row.
# - distances, indices = index.search(query_embedding, top_k): Performs the search on the FAISS index to find the top_k nearest neighbors, returning their distances and indices.
# - results = []: Initializes an empty list to store the search results.
# - for dist, idx in zip(distances[0], indices[0]): Iterates over the search results, unpacking the distances and indices.
# - if idx in doc_store: Checks if the index exists in the document store.
# - results.append(doc_store[idx]["chunk"]): If the index is found in the document store, appends the corresponding text chunk to the results.
# - logging.debug(f"Index {idx} not found in document store. Removing from FAISS index."): Logs a debug message if the index is not found in the document store.
# - index.remove_ids(faiss.IDSelectorRange(idx, idx + 1)): Removes the ID from the FAISS index to maintain consistency.
# - except Exception as e: Catches any exceptions that occur while removing the ID from the FAISS index.
# - logging.debug(f"Error removing ID {idx} from FAISS index: {e}"): Logs a debug message if an error occurs while removing the ID from the FAISS index.
# - return " ".join(results): Returns the concatenated text chunks of the top search results.


# Query llama.cpp
def query_llama_cpp(prompt, n_predict=200):
    """Query the llama.cpp model to generate a response based on a prompt.
    
    Args:
        prompt (str): The input prompt to generate a response from the model.
        n_predict (int): The number of tokens to predict/generate. Default is 200.
        
    Returns:
        str: The generated response from the model or an error message if the query fails.
    """
    try:
        # Construct the command to run the llama.cpp model with the specified arguments
        cmd = [
            r"D:\Projects\Chatbot\llama.cpp\bin\Release\llama-cli.exe",  # Path to the llama.cpp executable
            "--model",
            r"D:\Projects\Chatbot\llama.cpp\models\gemma-1.1-2b-it.Q3_K_M.gguf",  # Path to the model file
            "--prompt",
            prompt,  # The input prompt for the model
            "--n-predict",
            str(n_predict),  # Number of tokens to predict
        ]

        # Run the command using subprocess.run and capture the output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Return the generated response from the model, stripping any leading/trailing whitespace
        return result.stdout.strip()
    
    # Handle exceptions if the subprocess call fails
    except subprocess.CalledProcessError as e:
        # Log an error message with the exception details
        logging.error(f"Error querying llama.cpp: {e}")
        # Return an error message indicating that the response generation failed
        return "An error occurred while generating the response."

# Purpose: This function queries the llama.cpp model to generate a response based on a given prompt, handling potential errors gracefully.
# - try: Attempts to construct and run the command to query the llama.cpp model.
# - cmd: A list containing the command and arguments to run the llama.cpp model with the specified prompt and number of tokens to predict.
# - subprocess.run(cmd, capture_output=True, text=True, check=True): Runs the command and captures the output, ensuring the command is executed as a subprocess.
# - result.stdout.strip(): Returns the generated response from the model, stripping any leading or trailing whitespace.
# - except subprocess.CalledProcessError as e: Catches any exceptions that occur if the subprocess call fails.
# - logging.error(f"Error querying llama.cpp: {e}"): Logs an error message with the exception details.
# - return "An error occurred while generating the response.": Returns an error message indicating that the response generation failed if an exception occurs.


def answer_question(query):
    """Answer a user's question by searching the FAISS index and querying llama.cpp.
    
    Args:
        query (str): The user's query/question to be answered.
        
    Returns:
        str: The generated response from the llama.cpp model or an error message if the query fails.
    """
    # Search the FAISS index for relevant text based on the query
    context = search_faiss(query, top_k=3)
    
    # If no relevant context is found, return a message indicating that
    if not context:
        return "No relevant context found for your query."
    
    # Construct the full prompt by combining the question and the context
    full_prompt = f"Question: {query}\nAnswer:"
    
    # Record the start time for performance metrics
    start_time = time()
    
    # Query the llama.cpp model with the full prompt
    response = query_llama_cpp(full_prompt)
    
    # Calculate the elapsed time for generating the response
    elapsed_time = time() - start_time
    
    # Calculate the length of the response in tokens
    response_length = len(response.split())
    
    # Print performance metrics
    print(f"\nPerformance Metrics:")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Response length: {response_length} tokens")
    print(f"Query-processing rate: {response_length / elapsed_time:.2f} tokens/second\n")
    
    # Return the generated response
    return response

# Purpose: This function answers a user's question by searching the FAISS index for relevant context and querying the llama.cpp model to generate a response.
# - context = search_faiss(query, top_k=3): Searches the FAISS index for relevant text based on the query and retrieves the top 3 results.
# - if not context: Checks if no relevant context is found.
# - return "No relevant context found for your query.": Returns a message indicating that no relevant context was found.
# - full_prompt = f"Question: {query}\nAnswer:": Constructs the full prompt for the model by combining the question and the context.
# - start_time = time(): Records the start time for performance metrics.
# - response = query_llama_cpp(full_prompt): Queries the llama.cpp model with the full prompt to generate a response.
# - elapsed_time = time() - start_time: Calculates the elapsed time for generating the response.
# - response_length = len(response.split()): Calculates the length of the response in tokens.
# - print(f"\nPerformance Metrics:"): Prints the performance metrics for the response generation.
# - print(f"Time taken: {elapsed_time:.2f} seconds"): Prints the time taken to generate the response.
# - print(f"Response length: {response_length} tokens"): Prints the length of the response in tokens.
# - print(f"Query-processing rate: {response_length / elapsed_time:.2f} tokens/second\n"): Prints the query-processing rate in tokens per second.
# - return response: Returns the generated response from the llama.cpp model.


# List trained documents
def list_trained_documents():
    """Print a list of all trained documents.
    
    This function checks if there are any trained files. If there are, it prints the list of trained documents. If not, it prints a message indicating that no documents have been trained yet.
    """
    if trained_files:
        print("\nTrained Documents:")  # Print header for the list of trained documents
        for file in trained_files:
            # Print each trained file in the list
            print(f"- {file}")
    else:
        # Print a message indicating that no documents have been trained yet
        print("No documents have been trained yet.")

def interactive_menu():
    """Display an interactive menu for user actions.
    
    This function loads previously saved knowledge and displays an interactive menu with options for the user to list, train, remember, query, forget, or exit. It processes the user's choice and calls the corresponding functions.
    """
    load_knowledge()  # Load previously saved knowledge

    while True:
        print(MENU_BANNER)  # Display the menu banner
        print("1. List\n2. Train\n3. Remember\n4. Query\n5. Forget\n6. Exit")  # Display menu options
        choice = input("Enter your choice (1/2/3/4/5/6): ").strip()  # Get user input for menu choice

        if choice == "1":
            list_trained_documents()  # List all trained documents
        elif choice == "2":
            folder = input("Enter the folder path: ").strip()  # Prompt user for the folder path
            if os.path.isdir(folder):
                scan_and_process_files(folder)  # Scan and process files in the specified folder
                print("Training completed.")  # Print completion message
            else:
                print("Invalid folder path.")  # Print error message for invalid folder path
        elif choice == "3":
            save_knowledge()  # Save the current knowledge
        elif choice == "4":
            query = input("Enter your query: ").strip()  # Prompt user for a query
            print("\nProcessing query...")  # Print processing message
            answer = answer_question(query)  # Get the answer for the query
            print(f"{answer}")  # Print the answer
        elif choice == "5":
            flush_knowledge()  # Flush all knowledge
        elif choice == "6":
            print("Goodbye!")  # Print goodbye message
            break  # Exit the loop and end the program
        else:
            print("Invalid choice.")  # Print error message for invalid choice

if __name__ == "__main__":
    interactive_menu()  # Call the interactive menu function if the script is executed directly

# Purpose: These functions handle listing trained documents and providing an interactive menu for user actions.
# - list_trained_documents(): Prints a list of all trained documents. Checks if there are any trained files and prints them if they exist. Otherwise, it prints a message indicating that no documents have been trained yet.
# - interactive_menu(): Displays an interactive menu with options for the user to train, list, query, remember, forget, or exit. Loads previously saved knowledge and processes user input to call the corresponding functions.
# - load_knowledge(): Loads previously saved knowledge.
# - scan_and_process_files(folder): Scans and processes files in the specified folder for training.
# - answer_question(query): Processes the user's query and returns an answer.
# - save_knowledge(): Saves the current knowledge.
# - flush_knowledge(): Flushes all knowledge.
# - if __name__ == "__main__": Ensures that the interactive menu is called if the script is executed directly.
