
import os
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

with open("db/bm25_store.pkl", "rb") as f:
    bm25_chunks = pickle.load(f)
bm25_retriever = BM25Retriever.from_documents(bm25_chunks)
bm25_retriever.k=3

query = "how much amount did google agrees to pay to former employes in gender discrimination?"

dense_matches = vector_db.as_retriever(search_kwargs={"k": 3}).invoke(query)
sparse_matches = bm25_retriever.invoke(query)

combined_pool = list({doc.page_content: doc for doc in (dense_matches + sparse_matches)}.values())
print(f"Merged retrieval pool size: {len(combined_pool)} unique context blocks.")

print("Running FlashRank Cross-Encoder Re-scoring...")
reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=3)

final_docs = reranker.compress_documents(documents=combined_pool, query=query)

print(f"User Query: {query}")

print("--- Context ---")
print("\n--- Top Normalized Context Chunks ---")
for idx, doc in enumerate(final_docs, 1):
    print(f"Chunk {idx} [Relevance Score: {doc.metadata.get('relevance_score', 'N/A')}]:\n{doc.page_content[:150]}...\n")

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in final_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# 5. Connect to Cloud Acceleration (Groq)
print("Forwarding payload context to Llama-3.3 on Groq...")
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)


print("\n--- Generated Response ---")

print("Content only:")
print(result.content)
