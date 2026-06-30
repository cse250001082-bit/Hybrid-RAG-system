# 📘 Student Document Assistant (SDA)

## Problem Statement
Students and researchers frequently struggle to efficiently extract accurate insights from dense academic literature, lecture notes, and textbooks. Traditional keyword search tools often miss conceptual meaning, while generic LLMs are prone to hallucinations when asked about specific niche coursework material. Keeping up with a high volume of documents requires an intelligent system that can parse local files, preserve accurate context, and answer nuanced questions without leaking data to public model-training sets.

## Features
* **Hybrid Multi-Engine Retrieval:** Pairs semantic vector lookup (capturing abstract concepts) with BM25 lexical search (capturing exact keywords, formulas, and definitions).
* **Cross-Encoder Re-ranking:** Utilizes an integrated **FlashRank** layer to sort and filter raw search outputs, sending only high-confidence data to the language model.
* **Contextual Query Condensation:** Automatically restructures ongoing conversational history into clean, standalone search queries before scanning database storage.
* **Dynamic Sidebar Ingestion Pipeline:** Allows live file uploads (`.txt`, `.pdf`) directly through a web browser to build or wipe the local database indexing on the fly.
* **Transparent System Metrics:** Features a debugging expander that showcases real-time system health metrics, deduplication pool records, and FlashRank confidence scores.

## Tech Stack
* **User Interface:** Streamlit
* **Orchestration Framework:** LangChain (LangChain Core, Community, and Partner Extensions)
* **Vector Store Database:** ChromaDB (configured with Cosine Distance metrics)
* **Keyword Indexer:** BM25 Retriever
* **Embedding Model:** HuggingFace Embeddings (`all-MiniLM-L6-v2` running 100% locally)
* **Re-ranking Engine:** Flashrank Reranker (`ms-marco-MiniLM-L-12-v2`)
* **Inference Foundation Model:** ChatGroq (`llama-3.3-70b-versatile`)

## Project Architecture
The system functions through a distinct five-phase pipeline architecture:
1. **Ingestion:** Raw study materials ➔ Chunk Splitting ➔ Vectorization ➔ Synchronized Dual-Storage Serialization (ChromaDB + Pickle BM25 Store).
2. **Condensation:** User Question + Active Chat History ➔ Groq Standalone Search Query Generation.
3. **Hybrid Retrieval:** Search Query ➔ Parallel Dense Scan (ChromaDB) & Sparse Scan (BM25) ➔ Deduplicated Merge Pool.
4. **Re-ranking:** Merge Pool ➔ FlashRank Filter ➔ Top 3 Re-ranked Context Chunks.
5. **Generation:** Re-ranked Context Chunks + Prompt Wrapper ➔ Groq Streaming Engine ➔ UI Render.

## Installation

### 1. Environment Setup
Clone this repository and verify you are running Python 3.9+. Install the required dependencies:
```bash
pip install streamlit langchain-chroma langchain-huggingface langchain-community langchain-groq flashrank python-dotenv
2. Configure Environment Variables
Create a file named .env in the root folder of your workspace and insert your Groq API credentials:

Code snippet
GROQ_API_KEY=gsk_your_actual_api_key_here
Usage
Launching the Web Dashboard
Execute the following terminal command to run the interactive GUI application:

Bash
streamlit run final.py
Step 1: Drop textbook pages or lecture notes into the file uploader widget in the left sidebar panel.

Step 2: Click "Run Ingestion Pipeline" to compile the local knowledge indexes.

Step 3: Begin typing questions into the chat bar to converse with your study documents.

Running via Terminal CLI (Alternative)
To test the core architecture without launching a web server, drop text documents straight into a directory named docs/ and run:

Bash
python ingestion_pipeline.py      # Pre-processes the files
python retrieval_withchatmemory.py # Starts a conversational loop in your terminal
Future Scope
Document Parsing Upgrades: Integrating specialized OCR libraries like Unstructured or PyMuPDF to read structural elements like tables, graphs, and images from complex lecture PDFs.

Local LLM Foundations: Transitioning the cloud-based Groq inference layer to run completely local models via Ollama (e.g., Llama 3 or Mistral) for offline environments.

Advanced RAG Paradigms: Adding Parent-Document Retrieval structures to store small chunk segments for vector parsing while returning larger contextual text paragraphs to the LLM for generation.
