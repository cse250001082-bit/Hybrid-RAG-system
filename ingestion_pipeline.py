import warnings

warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*",
    category=DeprecationWarning
)

import os
import warnings
import pickle
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # To know wether docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # TO Load all .txt files from the docs directory
    loader = DirectoryLoader(
    docs_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db", bm25_path="db/bm25_store.pkl"):
    """Create and persist ChromaDB vector store using local HuggingFace embeddings"""
    print("Creating local embeddings and storing in ChromaDB...")
        
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # TO create chromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")

    print("--- Serializing BM25 Text Store ---")
    with open(bm25_path, "wb") as f:
        pickle.dump(chunks, f)
        
    print("✅ Dual-indexing complete. Subsystems ready for Hybrid lookup.")
    return vectorstore



def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")
    
    
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
   
    print("Persistent directory does not exist. Initializing vector store...\n")
    
         
    documents = load_documents(docs_path)  


    chunks = split_documents(documents)
    

    vectorstore = create_vector_store(chunks, persistent_directory)
    
    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__ == "__main__":
    main()
