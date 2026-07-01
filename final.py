import os
import pickle
import streamlit as st
from dotenv import load_dotenv


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


from ingestion_pipeline import load_documents, split_documents, create_vector_store

load_dotenv()

st.set_page_config(
    page_title="Student Document Assistant",
    page_icon="📘",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db_initialized" not in st.session_state:
    
    st.session_state.db_initialized = os.path.exists("db/chroma_db") and os.path.exists("db/bm25_store.pkl")

@st.cache_resource(show_spinner=False)
def init_retrieval_components():
    """Caches your model objects locally so they load instantly on button clicks."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True} if os.path.exists("db/chroma_db") else {}
    )
    
    vector_db = None
    bm25_retriever = None
    
    if st.session_state.db_initialized:
        vector_db = Chroma(
            persist_directory="db/chroma_db", 
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        with open("db/bm25_store.pkl", "rb") as f:
            bm25_chunks = pickle.load(f)
        bm25_retriever = BM25Retriever.from_documents(bm25_chunks)
        bm25_retriever.k = 3
        
    reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=3)
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    return embedding_model, vector_db, bm25_retriever, reranker, model

embedding_model, vector_db, bm25_retriever, reranker, model = init_retrieval_components()

with st.sidebar:
    st.title("⚙️ SDA Control Panel")
    st.markdown("---")
    
    st.subheader("📊 System Health Metrics")
    if st.session_state.db_initialized and vector_db is not None:
        doc_count = vector_db._collection.count()
        st.metric(label="ChromaDB Dense Records", value=f"{doc_count} Chunks")
        st.metric(label="BM25 Lexical Store", value="Active (k=3)")
        st.success("Subsystems Synchronized")
    else:
        st.warning("No Active Knowledge Base Found.")
        
    st.markdown("---")
    
    st.subheader("📂 Ingest Study Material")
    uploaded_files = st.file_uploader(
        "Upload Textbook PDFs or TXT notes:", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Run Ingestion Pipeline", use_container_width=True):
        if not uploaded_files:
            st.error("Please add files first!")
        else:
            with st.spinner("Executing Ingestion Layers..."):
                
                os.makedirs("docs", exist_ok=True)
                for f in uploaded_files:
                    with open(os.path.join("docs", f.name), "wb") as buffer:
                        buffer.write(f.getbuffer())
                
                raw_docs = load_documents(docs_path="docs")
                processed_chunks = split_documents(raw_docs)
                create_vector_store(processed_chunks)
                
                st.session_state.db_initialized = True
                st.cache_resource.clear()
                st.success("Indexing Complete! Reloading resources...")
                st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.title("📘 STUDENT DOCUMENT ASSISTANT")
st.caption("🚀 Advanced Hybrid Retrieval (Dense Vector + BM25 Lexical) with Local FlashRank Re-ranking")

if not st.session_state.db_initialized:
    st.info("👋 Welcome! To begin studying, drop your lecture documents or reference notes into the sidebar and run the ingestion pipeline.")
else:
   
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    if user_question := st.chat_input("Ask any question about your study materials:"):
        
        with st.chat_message("user"):
            st.markdown(user_question)
            
        if st.session_state.chat_history:
            history_prompt = [
                SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")
            ] + st.session_state.chat_history + [HumanMessage(content=f"New question: {user_question}")]
            
            search_question = model.invoke(history_prompt).content.strip()
        else:
            search_question = user_question
            
        dense_hits = vector_db.as_retriever(search_kwargs={"k": 3}).invoke(search_question)
        sparse_hits = bm25_retriever.invoke(search_question)
        
        merged_pool = list({doc.page_content: doc for doc in (dense_hits + sparse_hits)}.values())
        
        refined_chunks = reranker.compress_documents(documents=merged_pool, query=search_question)
        
        context_payload = "\n".join([f"- {doc.page_content}" for doc in refined_chunks])
        combined_input = f"""Based on the following documents, please answer this question: {user_question}

        Documents:
        {context_payload}

        Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
        """
        
        messages = [
            SystemMessage(content="You are a helpful academic assistant answering questions strictly based on the provided references."),
        ] + st.session_state.chat_history + [HumanMessage(content=combined_input)]
        
        with st.chat_message("assistant"):
           
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in model.stream(messages):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
           
            response_placeholder.markdown(full_response)
            
            with st.expander("🔍 See Live Pipeline Verification Metrics"):
                st.markdown(f"**Standalone Search Optimization Output:** `{search_question}`")
                st.markdown(f"**Merged Retrieval Pool Count:** `{len(merged_pool)} Chunks` -> **Normalized to Top:** `3 Chunks`")
                for i, doc in enumerate(refined_chunks, 1):
                    score = doc.metadata.get('relevance_score', 'N/A')
                   
                    if isinstance(score, float):
                        score = f"{score:.4f}"
                    st.markdown(f"**Chunk {i} [FlashRank Confidence Score: {score}]:**")
                    st.caption(f"Source: {doc.metadata.get('source', 'Unknown')} | Text preview: {doc.page_content[:200]}...")
        
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=full_response))
