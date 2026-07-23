# 

import os
import json
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini Models
print("Initializing Gemini...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

class ChunkMetadata(BaseModel):
    summary: str = Field(description="A short summary of the chunk's content")
    hypothetical_questions: list[str] = Field(description="2-3 hypothetical questions this chunk could answer")

def process_document_with_checkpoints(filepath):
    print(f"\nProcessing {filepath}...")
    out_file = f"{os.path.basename(filepath)}_processed.json"
    
    # --- CHECKPOINT LOGIC ---
    # Load existing progress if the file already exists
    processed_chunks = []
    processed_ids = set()
    
    if os.path.exists(out_file):
        with open(out_file, "r", encoding='utf-8') as f:
            try:
                processed_chunks = json.load(f)
                processed_ids = {chunk["chunk_id"] for chunk in processed_chunks}
                print(f"Loaded {len(processed_chunks)} previously processed chunks. Resuming...")
            except json.JSONDecodeError:
                print("Existing JSON is corrupted. Starting fresh.")

    # --- CHUNKING LOGIC ---
    with open(filepath, 'r', encoding='utf-8') as f:
        text_content = f.read()
        
    from langchain_core.documents import Document
    doc = Document(page_content=text_content, metadata={"source": filepath})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents([doc])
    
    print(f"Total structurally-aware chunks found: {len(chunks)}.")
    
    structured_llm = llm.with_structured_output(ChunkMetadata)
    
    # --- PROCESSING LOGIC ---
    for i, chunk in enumerate(chunks):
        current_chunk_id = f"{os.path.basename(filepath)}_chunk_{i}"
        
        # Skip if already processed in a previous run
        if current_chunk_id in processed_ids:
            continue
            
        print(f"Generating metadata for chunk {i+1} of {len(chunks)}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                metadata_str = " | ".join([f"{k}: {v}" for k, v in chunk.metadata.items()])
                prompt = f"Analyze the following text from the OmniTech Q3 Operations Report (Context: {metadata_str}) and provide a brief summary and 2-3 hypothetical questions it answers.\n\nText:\n{chunk.page_content}"
                
                result = structured_llm.invoke(prompt)
                
                new_chunk_data = {
                    "chunk_id": current_chunk_id,
                    "text": chunk.page_content,
                    "document_structure": chunk.metadata,
                    "summary": result.summary,
                    "questions": result.hypothetical_questions
                }
                
                processed_chunks.append(new_chunk_data)
                
                # SAVE PROGRESS IMMEDIATELY AFTER EVERY CHUNK
                with open(out_file, "w", encoding='utf-8') as f:
                    json.dump(processed_chunks, f, indent=2)
                    
                print(f"  ✓ Chunk {i+1} saved.")
                time.sleep(4) # Respect the 15 RPM free tier limit
                break 
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    wait_time = 60 * (attempt + 1)
                    print(f"  [RATE LIMIT] Quota exceeded on attempt {attempt+1}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error processing chunk {i+1}: {e}")
                    break 

if __name__ == "__main__":
    target_file = "OmniTech_Q3_Financial_and_Strategic_Operations_Report.txt"
    if os.path.exists(target_file):
        process_document_with_checkpoints(target_file)
    else:
        print(f"File {target_file} not found.")