import os
import json
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import pickle

# Load API keys
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY is not set.")
    exit(1)

def build_vector_database():
    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # We will store the database locally in a folder named 'chroma_db'
    db_dir = "./chroma_db"
    
    # Initialize Chroma client
    client = chromadb.PersistentClient(path=db_dir)
    collection_name = "sec_10k_collection"
    
    print(f"Connecting to ChromaDB at {db_dir}...")
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    
    # Load the processed JSON data
    data_file = "OmniTech_Q3_Financial_and_Strategic_Operations_Report.txt_processed.json"
    if not os.path.exists(data_file):
        print(f"Error: Processed data file {data_file} not found. Did the ingest script finish?")
        return
        
    print(f"Loading chunks from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        
    print(f"Found {len(chunks_data)} processed chunks.")
    
    documents = []
    
    for item in chunks_data:
        # We enrich the chunk text with the hypothetical questions and summary so that vector
        # search will match on not just the raw text, but also the generated questions
        enriched_page_content = f"Summary: {item['summary']}\nQuestions This Answers: {', '.join(item['questions'])}\n\nDocument Text:\n{item['text']}"
        
        # We store the structural metadata cleanly back into the Document
        metadata = item.get('document_structure', {})
        metadata['chunk_id'] = item['chunk_id']
        metadata['source'] = "OmniTech_Q3"
        
        doc = Document(
            page_content=enriched_page_content,
            metadata=metadata
        )
        documents.append(doc)
        
    print("Building BM25 Keyword Search Index...")
    # This runs purely locally on the text, no API calls!
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 2  # Retrieve top 2 results
    
    # Save the BM25 index locally using pickle
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("   ✓ Saved BM25 Index to bm25_index.pkl")
        
    print("Inserting chunks into vector database (this might take a minute)...")
    # Add documents in batches to avoid overwhelming the system
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(documents=batch)
        print(f"   ✓ Inserted documents {i+1} to {min(i+batch_size, len(documents))}...")
        
    print(f"\nSuccessfully built Vector Database '{collection_name}' with {len(documents)} total chunks!")
    
    # Let's do a quick test retrieval
    print("\n--- Running a Test Hybrid Query ---")
    query = "Why did the European sector revenue plummet?"
    print(f"Q: {query}")
    
    print("\n1. Semantic Search (ChromaDB + Gemini Embeddings):")
    results = vector_store.similarity_search_with_score(query, k=2)
    for res, score in results:
        head = res.metadata.get('source', 'Unknown Section')
        print(f"  Found match in Document: {head} (Score: {score:.4f})")
        print(f"  Preview: {res.page_content[:150].replace(chr(10), ' ')}...")
        
    print("\n2. Keyword/Exact Search (Local BM25):")
    bm25_results = bm25_retriever.invoke(query)
    for res in bm25_results:
        head = res.metadata.get('source', 'Unknown Section')
        print(f"  Found exact keyword match in Document: {head}")
        print(f"  Preview: {res.page_content[:150].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    build_vector_database()
