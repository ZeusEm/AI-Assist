#AI Assist - Comprehensive Documentation 🧠🤖
=======================================

Welcome to the documentation for **AI Assist**, a sophisticated Python-based chatbot that leverages the power of natural language processing and machine learning to provide insightful and contextually aware responses. This document will guide you through the process of setting up, using, and understanding the repository.

🔍 Key Features
------------

*   **Multimodal File Support:** Train the AI on text extracted from PDFs, Microsoft Word documents (`.docx`), and image files (including `.png`, `.jpg`, `.jpeg`, `.bmp`, and `.tiff` formats).
*   **Efficient File Processing:** Hashing ensures only new or modified files are retrained, reducing redundant computations.
*   **FAISS Integration:** FAISS index ensures fast and efficient retrieval of relevant context for queries.
*   **Extensive Logging:** Detailed logs provide clarity on application processes and debugging.

🛠️ Technology Stack
----------------

*   **Python:** The core programming language.
*   **Sentence Transformers:** `sentence-transformers==2.2.2` for semantic embeddings.
*   **FAISS:** `faiss-cpu==1.7.3` for efficient indexing and retrieval.
*   **PDFPlumber:** `pdfplumber==0.5.28` for PDF text extraction.
*   **Python-Docx:** `python-docx==0.8.11` for handling Word documents.
*   **Pillow:** `pillow==9.4.0` for image preprocessing.
*   **PyTesseract:** `pytesseract==0.3.10` for OCR on image files.
*   **TQDM:** `tqdm==4.64.1` for progress visualization.

🔧 Process Flow
------------

1.  **User Interaction:** The user selects from options to train, list documents, query, save knowledge, flush knowledge, or exit.
2.  **Document Processing:** Based on the selected mode, the system identifies supported files in the given folder (`.pdf`, `.docx`, images) and extracts text.
3.  **Hashing:** Each file is hashed to determine if it is new or modified since the last training.
4.  **Training:** Semantic embeddings are generated using Sentence Transformers and added to the FAISS index.
5.  **Query Handling:** The system retrieves relevant chunks from the FAISS index and constructs a prompt for the Llama.cpp model to generate a response.
6.  **Response Generation:** The chatbot delivers the generated response to the user.

🔨 Setup Instructions
------------------

### 1\. Prerequisites

*   Ensure Python 3.8 or higher is installed.
*   Install Anaconda (recommended for managing dependencies).
*   Install Visual Studio Build Tools from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
*   Install CMake from [here](https://cmake.org/download/).

### 2\. Repository Clone

    git clone https://github.com/ZeusEm/AI-Assist.git
    cd AI-Assist

### 3\. Update Batch File

Edit `AI Assist.bat` to point to your Python executable and script location:

    "C:\Path\to\python.exe" "path\to\ai_assist.py" %*

### 4\. Install Dependencies

    
    pip install sentence-transformers
    pip install pdfplumber
    pip install python-docx
    pip install faiss-cpu
    pip install pytesseract
    pip install pillow
    pip install tqdm
    

### 5\. Download Llama Model

Download the `gemma-1.1-2b-it-Q3_K_M.gguf` model from [here](https://huggingface.co/bartowski/gemma-1.1-2b-it-GGUF) and place it in the `llama.cpp/models` directory.

### 6\. Build Llama.cpp

1.  Clone and extract [Llama.cpp](https://github.com/ggerganov/llama.cpp).
2.  Navigate to the directory:

    cd path\to\llama.cpp

4.  Run the following commands to build:

    cmake .
    cmake --build . --config Release

### 7\. Update Paths

Edit `ai_assist.py` to point to:

    LLAMA_CLI_PATH = "path\to\llama.cpp\bin\Release\llama-cli.exe"
    MODEL_PATH = "path\to\llama.cpp\models\gemma-1.1-2b-it.Q3_K_M.gguf"

🚀 Usage
-----

Run the `AI Assist.bat` file to launch the application. Follow the menu options to train, query, or manage knowledge.

🛡️ Key Advantages
------------

*   **Efficiency**: Skips unchanged files during training using hashing.
*   **Accuracy**: Employs high-quality embeddings and contextual reasoning for precise answers.
*   **Scalability**: Capable of handling large datasets.
*   **Flexibility**: Easily adaptable for diverse use cases.

🤝 Contributing
------------

We welcome contributions! Feel free to fork the repository, make changes, and submit a pull request.

📋 License
-------

This project is licensed under the GNU GPL v3 License.
