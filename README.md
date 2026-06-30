# 📘 Student Document Assistant (SDA)

An advanced Retrieval-Augmented Generation (RAG) system engineered with a **Hybrid Search** core (Dense Vector + Lexical BM25) and local **FlashRank Re-ranking**. This application allows students to upload lecture notes, textbook chapters, or reference material and converse with their documents through a clean, conversational Streamlit UI.

---

## 🚀 Key Architectural Features

* **Dual-Engine Hybrid Search:** Combines semantic context lookup (**ChromaDB** with cosine distance) and keyword matching (**BM25 Retriever**) to capture both concepts and exact terms.
* **Context Re-ranking:** Utilizes a local cross-encoder (`ms-marco-MiniLM-L-12-v2` via **FlashRank**) to filter and re-rank retrieved text pieces, ensuring high-density relevancy passes into the LLM.
* **Conversational Memory with De-contextualization:** Injects an isolated Llama-3 step that rewrites user follow-ups into standalone queries based on prior chat logs before hitting the search databases.
* **Production Ingestion Pipeline:** Splits incoming `.txt` or `.pdf` data natively, tracking source metadata footprints seamlessly.
* **Inference Speed:** Powered by `llama-3.3-70b-versatile` hosted on **Groq** for high-velocity streaming token generation.

---

## 🛠️ System Architecture Flow

1. **Ingestion Layer:** Raw Documents (`docs/`) ➔ Character Splitting ➔ Vector Generation (`all-MiniLM-L6-v2`) ➔ ChromaDB & Serialized BM25 Storage.
2. **Retrieval Layer:** Conversational Query Condensation ➔ Parallel Dense + Sparse Retrieval ➔ Deduplication Pool ➔ FlashRank Re-ranking.
3. **Generation Layer:** Augmented Prompt Construction ➔ Groq Llama-3.3 Cloud Streaming ➔ Streamlit UI Rendering.

---

## 📁 Repository Blueprint

```text
├── db/                        # Local database persistence (Chroma & Pickle)
│   ├── chroma_db/             # Vector database directory
│   └── bm25_store.pkl         # Serialized keyword chunks
├── docs/                      # Temporary storage for uploaded raw text/PDF assets
├── ingestion_pipeline.py      # Independent data vectorization layer
├── retrieval_withchatmemory.py# Terminal CLI application supporting stateful chat
├── final.py                   # Production Streamlit UI Dashboard
├── .env                       # Local secrets configuration file
└── README.md                  # System documentation
⚡ Quick Start
1. Prerequisites & Environment Setup
Clone this repository and ensure you have Python 3.9+ installed. Create your virtual environment and install the required distribution libraries:

Bash
pip install streamlit langchain-chroma langchain-huggingface langchain-community langchain-groq flashrank python-dotenv
2. Configure Credentials
Create a .env file in the root directory and add your Groq API key:

Code snippet
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
3. Execution Options
You can run this application through two different interfaces:

Option A: Production Web Interface (Recommended)
Launch the Streamlit web dashboard to upload files dynamically, view system health metrics, and trace pipeline validation live:

Bash
streamlit run final.py
Option B: Terminal Command Line Chat
If you prefer testing inside your terminal with pre-existing files located inside the docs/ folder, execute the baseline ingestion pipeline followed by the chat engine:

Bash
# 1. Parse documents first
python ingestion_pipeline.py

# 2. Start the interactive terminal conversation loop
python retrieval_withchatmemory.py
🔍 Visual Verification Analytics
The Streamlit UI includes an expandable Pipeline Verification Metrics drawer. When chatting with your documents, this lets you inspect:

The optimized standalone variation of your question.

FlashRank confidence scores indicating how strongly individual chunks relate to your query.

Metadata trails verifying exactly what source document or page section fed the answer.
