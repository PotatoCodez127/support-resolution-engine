import chromadb
from chromadb.utils import embedding_functions

def ingest_documents():
    print("🚀 Initializing Vector Database...")
    
    # 1. Initialize ChromaDB (stores data locally in a folder called 'chroma_db')
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # 2. Use a fast, local embedding model to convert text to vectors
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # 3. Create or get our policy collection
    collection = chroma_client.get_or_create_collection(
        name="company_policies",
        embedding_function=sentence_transformer_ef
    )
    
    print("📄 Reading company policies...")
    with open("company_policies.txt", "r") as file:
        raw_text = file.read()
        
    # 4. Chunking: We split the text by "SECTION" so each policy rule is its own searchable chunk.
    # In production with PDFs, you'd use a tool like RecursiveCharacterTextSplitter.
    chunks = raw_text.split("SECTION")
    
    # Clean up chunks and ignore empty ones
    valid_chunks = ["SECTION" + chunk.strip() for chunk in chunks if chunk.strip()]
    
    # 5. Prepare data for the Vector DB
    documents = []
    ids = []
    
    for i, chunk in enumerate(valid_chunks):
        documents.append(chunk)
        ids.append(f"policy_chunk_{i}")
        
    print(f"🧩 Storing {len(documents)} chunks into the vector database...")
    
    # 6. Upsert (Insert or Update) into ChromaDB
    collection.upsert(
        documents=documents,
        ids=ids
    )
    
    print("✅ Ingestion complete! The AI is now ready to read from the Vector DB.")

if __name__ == "__main__":
    ingest_documents()