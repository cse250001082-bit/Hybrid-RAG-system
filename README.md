# 📘 Student Document Assistant

## Problem Statement

Students often need to search through lengthy lecture notes, textbooks, and study materials to find specific information. Traditional keyword search fails to understand the meaning of queries and cannot handle conversational follow-up questions effectively.

This project develops an AI-powered Student Document Assistant using Retrieval-Augmented Generation (RAG). It combines semantic search, keyword search, reranking, and conversational memory to provide accurate answers directly from uploaded study materials.

---

## Features

- 📂 Upload multiple text or PDF study materials.
- 🔍 Hybrid Retrieval using:
  - Dense Vector Search (ChromaDB)
  - Sparse Lexical Search (BM25)
- ⚡ FlashRank reranking for highly relevant document retrieval.
- 🧠 Conversational memory for follow-up questions.
- 🤖 AI-generated responses using Groq's Llama-3.3-70B model.
- 📊 Displays retrieval statistics and confidence scores.
- 💾 Persistent vector database using ChromaDB.
- 🌐 Interactive web interface built with Streamlit.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Interface |
| LangChain | RAG Pipeline |
| ChromaDB | Vector Database |
| HuggingFace Embeddings | Text Embeddings |
| BM25 Retriever | Keyword-based Retrieval |
| FlashRank | Document Re-ranking |
| Groq API | Large Language Model |
| dotenv | Environment Variable Management |

---

## Project Architecture

```
                    User
                      │
                      ▼
             Streamlit Interface
                      │
                      ▼
            User Uploads Documents
                      │
                      ▼
            Ingestion Pipeline
      ┌─────────────────────────┐
      │ Load Documents          │
      │ Split into Chunks       │
      │ Generate Embeddings     │
      │ Store in ChromaDB       │
      │ Save BM25 Index         │
      └─────────────────────────┘
                      │
                      ▼
               User Question
                      │
                      ▼
         Chat History Condensation
                      │
                      ▼
        Hybrid Retrieval (Dense + BM25)
                      │
                      ▼
            FlashRank Re-ranking
                      │
                      ▼
           Context-Augmented Prompt
                      │
                      ▼
          Groq Llama-3.3-70B Model
                      │
                      ▼
              AI Generated Answer
```

### Project Files

- **final.py**
  - Main Streamlit application
  - Handles document upload
  - User interface
  - Chat interaction
  - Hybrid retrieval pipeline

- **ingestion_pipeline.py**
  - Loads documents
  - Splits text into chunks
  - Generates embeddings
  - Creates ChromaDB vector database
  - Stores BM25 index

- **retrieval_withmemory.py**
  - Retrieves relevant documents
  - Maintains conversation history
  - Performs hybrid search
  - Uses FlashRank reranking
  - Generates AI responses

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/student-document-assistant.git
cd student-document-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

```text
GROQ_API_KEY=your_api_key
```

### 5. Run the application

```bash
streamlit run final.py
```

---

## Usage

1. Launch the Streamlit application.
2. Upload one or more study documents.
3. Click **Run Ingestion Pipeline**.
4. Wait for the documents to be indexed.
5. Ask questions related to the uploaded materials.
6. The system:
   - Retrieves relevant document chunks.
   - Performs hybrid retrieval (Dense + BM25).
   - Re-ranks results using FlashRank.
   - Generates answers using the Groq LLM.
7. Continue asking follow-up questions—the assistant remembers previous conversation context.

---

## Future Scope

- Support additional document formats (DOCX, PPTX, CSV, HTML).
- OCR support for scanned PDFs and handwritten notes.
- Voice-based question answering.
- Multi-user authentication and personalized document collections.
- Cloud deployment using Docker and Kubernetes.
- Citation generation with page references.
- Integration with Learning Management Systems (LMS).
- Support for multilingual document retrieval.
- Advanced analytics and document summarization.
- Real-time collaboration and shared knowledge bases.
