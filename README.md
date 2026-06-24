# 📚 Multi-PDF RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) application that enables users to upload multiple PDF documents and interact with them through natural language conversations. The system retrieves relevant information from uploaded documents using semantic search and generates context-aware responses using Llama 3.3.

## Features

* Upload and query multiple PDF documents
* Conversational chat interface
* Retrieval-Augmented Generation (RAG) pipeline
* Semantic search using vector embeddings
* Multi-query retrieval for improved context retrieval
* Context-aware responses using chat history
* FastAPI backend with Streamlit frontend
* ChromaDB vector storage

## Tech Stack

### Frontend

* Streamlit

### Backend

* FastAPI
* Pydantic

### AI / LLM

* LangChain
* Meta Llama 3.3 70B Instruct
* Hugging Face Inference Endpoints

### Embeddings & Retrieval

* BAAI/bge-base-en-v1.5
* ChromaDB
* MultiQueryRetriever

### Document Processing

* PyPDFLoader
* RecursiveCharacterTextSplitter

## System Architecture

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
Text Chunking
(RecursiveCharacterTextSplitter)
    ↓
Embeddings
(BGE Base v1.5)
    ↓
ChromaDB
(Vector Database)
    ↓
MultiQueryRetriever
    ↓
Relevant Context Retrieval
    ↓
Llama 3.3 70B
(Hugging Face Endpoint)
    ↓
Generated Answer
```

## How It Works

1. Users upload one or more PDF documents.
2. PDFs are parsed and split into smaller chunks.
3. Text chunks are converted into vector embeddings.
4. Embeddings are stored in ChromaDB.
5. User questions are expanded into multiple semantic variations using MultiQueryRetriever.
6. Relevant document chunks are retrieved from the vector database.
7. Retrieved context and chat history are passed to Llama 3.3.
8. The model generates a grounded response based on the uploaded documents.

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd multi-pdf-rag-chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## Run Backend

```bash
uvicorn main:app --reload
```

Backend will run on:

```text
http://localhost:8000
```

## Run Frontend

```bash
streamlit run streamlit_app_ui.py
```

Frontend will run on:

```text
http://localhost:8501
```

## API Endpoints

### Upload PDFs

```http
POST /upload
```

Uploads and indexes one or more PDF documents.

### Ask Questions

```http
POST /ask
```

Request Body:

```json
{
  "question": "What is the document about?",
  "chat_history": "Previous conversation history"
}
```

## Future Improvements

* Persistent vector database storage
* Document source citations
* Hybrid search (Keyword + Semantic Search)
* User authentication
* Cloud deployment support
* Conversation memory management
* Reranking for improved retrieval quality

## Project Highlights

* Built a complete Retrieval-Augmented Generation (RAG) pipeline
* Implemented semantic document retrieval using vector embeddings
* Enhanced retrieval quality using MultiQueryRetriever-based query expansion
* Integrated Llama 3.3 for context-aware answer generation
* Developed a full-stack AI application using FastAPI and Streamlit

## License

This project is intended for educational and portfolio purposes.
