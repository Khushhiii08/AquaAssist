import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Dynamically resolve project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

JSON_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "chunks.json")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 100

def build_database():
    print(f"[*] Project root: {PROJECT_ROOT}")
    print(f"[*] Loading {MODEL_NAME} locally. This may take a minute on initial download...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"[*] Initializing local ChromaDB at: {DB_PATH}")
    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    collection = client.get_or_create_collection(name="aquaculture_knowledge")
    
    print(f"[*] Loading processed chunks from {JSON_PATH}...")
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"[-] Error: {JSON_PATH} not found. Run ingest.py first.")
        return

    total_chunks = len(chunks)
    print(f"[+] Found {total_chunks} chunks to embed.")

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        ids = [item["id"] for item in batch]
        texts = [item["text"] for item in batch]
        metadatas = [item["metadata"] for item in batch]
        
        print(f"    -> Embedding batch {i} to {i + len(batch)}...")
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print("\n[*] SUCCESS: Vector database successfully built!")
    print(f"[*] Stored at: {DB_PATH}")

if __name__ == "__main__":
    build_database()