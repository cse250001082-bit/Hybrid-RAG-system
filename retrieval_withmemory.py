import pickle
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
    )
vector_db = Chroma(
    persist_directory=persistent_directory, 
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
    )
with open("db/bm25_store.pkl", "rb") as f:
    bm25_chunks = pickle.load(f)
bm25_retriever = BM25Retriever.from_documents(bm25_chunks)
bm25_retriever.k=3

# Set up AI model
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
     # Step 2: Extract Hybrid Search Chunks
    dense_hits = vector_db.as_retriever(search_kwargs={"k": 3}).invoke(search_question)
    sparse_hits = bm25_retriever.invoke(search_question)
    
    # Merge keeping document items unique
    merged_pool = list({doc.page_content: doc for doc in (dense_hits + sparse_hits)}.values())

    reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=3)
     # Step 3: Run Local FlashRank Filter 
    refined_chunks = reranker.compress_documents(documents=merged_pool, query=search_question)

    print(f"User Query: {search_question}")
    # Display results
    print("--- Context ---")
    print("\n--- Top Normalized Context Chunks ---")
    for idx, doc in enumerate(refined_chunks, 1):
        print(f"Chunk {idx} [Relevance Score: {doc.metadata.get('relevance_score', 'N/A')}]:\n{doc.page_content[:150]}...\n")

    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in refined_chunks])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()
