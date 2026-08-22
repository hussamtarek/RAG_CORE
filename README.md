# RAG_CORE
A Retrieval-Augmented Generation (RAG) core system designed and optimized for cardiovascular and cardiac research applications.
RAG CORE 🫀

RAG CORE is a Retrieval-Augmented Generation (RAG) system designed to provide grounded and reliable information about cardiovascular diseases.

The system retrieves relevant information from trusted medical documents and uses an LLM to generate answers based on the retrieved context.

🚀 Features

* 📚 PDF document ingestion and processing
* 🔎 Semantic search using vector embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* 🎯 Relevance-based document retrieval
* 💬 Interactive Streamlit chat interface
* 🌍 Multi-language support
* 📝 Conversation history
* 📖 Source-based answers
* ⚙️ Configurable retrieval settings

🛠️ Tech Stack

* Python
* Streamlit
* ChromaDB
* Sentence Transformers
* BAAI/bge-m3
* Groq API
* Qwen
* Unstructured
* SQLite

📂 Project Structure

RAG-CORE/
│
├── app1.py
├── retrieval3.py
├── ingestion.py
├── requirements.txt
├── README.md
│
├── data/
│   └── medical PDF documents
│
├── assets/
│   └── logo.png
│
├── chroma_db/
│   └── vector database
│
└── chat_history.db

⚙️ Installation

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG-CORE

2. Create a virtual environment

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

🔑 API Key Setup

RAG CORE requires a Groq API key.

Create an environment variable called:

GROQ_API_KEY

On Windows PowerShell:

$env:GROQ_API_KEY="YOUR_API_KEY"

Or create a .env file:

GROQ_API_KEY=YOUR_API_KEY

Never upload your API key to GitHub.

📥 Data Ingestion

Place the required PDF documents inside:

data/

Then run the ingestion script:

python ingestion.py

The system will process the documents, split them into chunks, generate embeddings, and store them in ChromaDB.

🧠 Embedding Model

RAG CORE uses:

BAAI/bge-m3

The embedding model is used to convert document chunks and user queries into vector representations for semantic retrieval.

🔍 Retrieval Pipeline

The RAG pipeline works through the following steps:

User Question
      ↓
Language Detection
      ↓
Query Processing
      ↓
Embedding Generation
      ↓
Vector Search
      ↓
Relevant Chunks
      ↓
Relevance Filtering
      ↓
Best Context Selection
      ↓
LLM Generation
      ↓
Grounded Answer

▶️ Run the Application

After completing the installation and ingestion steps, run:

streamlit run app1.py

The application will open in your browser.

💬 Using RAG CORE

1. Enter your question in the chat interface.
2. The system processes the query.
3. Relevant information is retrieved from the medical knowledge base.
4. The most relevant context is passed to the language model.
5. The model generates a grounded response.
6. The conversation can be preserved through chat history.

⚙️ Retrieval Configuration

The system supports configurable retrieval parameters such as:

* Top K retrieved chunks
* Relevance threshold
* Final context selection
* Chunk size
* Chunk overlap

Current configuration includes:

Embedding Model: BAAI/bge-m3
Chunk Size: 1000
Chunk Overlap: 400
Top K: 5
Final Top K: 1
Distance Metric: Cosine

🧩 Main Components

Ingestion

Responsible for:

* Loading PDF documents
* Parsing document content
* Splitting documents into chunks
* Generating embeddings
* Storing vectors in ChromaDB

Retrieval

Responsible for:

* Processing user queries
* Generating query embeddings
* Searching the vector database
* Calculating relevance
* Selecting the best context

Generation

The retrieved context is provided to the LLM to generate an answer grounded in the available documents.

Streamlit Interface

The Streamlit application provides:

* Interactive chat
* Language selection
* Retrieval configuration
* Conversation history
* Source information
* User-friendly interface

⚠️ Medical Disclaimer

RAG CORE is an educational and informational project.

It is not a replacement for professional medical advice, diagnosis, or treatment.

Users should consult qualified healthcare professionals for medical decisions.

🔐 Security

Do not commit sensitive information such as:

.env
API keys
Passwords
Private credentials

Add sensitive files to .gitignore.

Example:

.env
__pycache__/
*.pyc
venv/
chat_history.db

🎯 Project Goal

The goal of RAG CORE is to demonstrate how Retrieval-Augmented Generation can be used to build a more grounded and reliable AI assistant by combining:

Document Retrieval + Semantic Search + Large Language Models

👥 Team

RAG CORE was developed as a collaborative AI project.

⸻

⭐ If you find this project useful, consider giving the repository a star!
