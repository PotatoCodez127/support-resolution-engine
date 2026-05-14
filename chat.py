import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from ollama import Client

# Load environment variables
load_dotenv()

# Initialize Ollama Client pointing to the Cloud API
client = Client(
    host='https://ollama.com',
    headers={'Authorization': f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
)

def retrieve_context(query: str) -> str:
    """
    Searches the local ChromaDB vector database for the most relevant policy chunks.
    """
    # 1. Connect to our existing local database
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # 2. Use the exact same embedding model we used during ingestion!
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = chroma_client.get_collection(
        name="company_policies",
        embedding_function=sentence_transformer_ef
    )
    
    # 3. Query the DB. n_results=2 means we want the top 2 most relevant policy sections.
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    
    # Combine the retrieved chunks into a single string to feed to the LLM
    retrieved_documents = results['documents'][0]
    return "\n\n--- NEXT POLICY CHUNK ---\n\n".join(retrieved_documents)

def answer_query(user_query: str):
    """
    Uses the RAG pattern to answer the user's query based strictly on retrieved context.
    """
    print("🔍 Searching knowledge base for relevant policies...")
    
    # Step A: Retrieve the relevant context from the Vector DB
    context = retrieve_context(user_query)
    
    print("🧠 Context retrieved. Generating strict response...\n")
    
    # Step B: Build the strict System Prompt. 
    # This is the "Open Book Exam" instructions for the LLM.
    system_prompt = f"""
    You are a polite customer support agent for ACME Electronics. 
    You MUST answer the user's question using ONLY the knowledge context provided below.
    
    CRITICAL RULES:
    1. If the answer is not explicitly stated in the context, say "I'm sorry, but I don't have that information in my current policy documents."
    2. Do NOT invent policies, guess, or use outside internet knowledge.
    3. Keep your answer concise and helpful.
    
    KNOWLEDGE CONTEXT:
    {context}
    """
    
    # Step C: Call the LLM
    response = client.chat(
        model="gemma4:31b-cloud", # Swap this to whatever model you used in the last step
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )
    
    print(f"🤖 AGENT:\n{response['message']['content']}\n")
    print("=" * 60)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Welcome to the ACME Support Center (RAG Enabled)")
    print("Type 'exit' to quit.")
    print("="*60 + "\n")
    
    while True:
        user_input = input("👤 YOU: ")
        if user_input.lower() == 'exit':
            break
        answer_query(user_input)