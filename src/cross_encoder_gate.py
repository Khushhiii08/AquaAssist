import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# Configuration
DB_PATH = "./data/chroma_db/"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CROSS_ENCODER_MODEL = "cross-encoder/nli-deberta-v3-small"
ENTAILMENT_THRESHOLD = 0.85

def initialize_gate():
    print(f"[*] Loading Embedding model ({EMBEDDING_MODEL_NAME})...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"[*] Loading Cross-Encoder model ({CROSS_ENCODER_MODEL})...")
    ce_model = CrossEncoder(CROSS_ENCODER_MODEL)
    
    print(f"[*] Connecting to local ChromaDB at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    collections = client.list_collections()
    if not collections:
        raise ValueError("[-] Error: The chroma_db folder contains no collections.")
        
    collection_name = collections[0].name
    print(f"[+] Successfully bound to collection: '{collection_name}'")
    collection = client.get_collection(name=collection_name)
    
    return embed_model, ce_model, collection

def evaluate_and_filter_chunks(query, embed_model, ce_model, collection, top_k_retrieve=15):
    print(f"\n[?] Query: '{query}'")
    print(f"[*] Step 1: Encoding query with {EMBEDDING_MODEL_NAME} (1024-dim)...")
    
    # Manually encode query to match BGE-M3 dimensions
    query_embedding = embed_model.encode([query], normalize_embeddings=True).tolist()
    
    print(f"[*] Step 2: Retrieving top-{top_k_retrieve} candidate chunks from ChromaDB...")
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k_retrieve
    )
    
    retrieved_texts = results['documents'][0]
    retrieved_ids = results['ids'][0]
    retrieved_metadatas = results['metadatas'][0]
    
    # Construct pairs for the Cross-Encoder: [Premise (Query), Hypothesis (Chunk)]
    pairs = [[query, text] for text in retrieved_texts]
    
    print(f"[*] Step 3: Running NLI Cross-Encoder verification gate...")
    predictions = ce_model.predict(pairs)
    
    verified_chunks = []
    
    print(f"[*] Step 4: Filtering chunks against threshold (>= {ENTAILMENT_THRESHOLD})...")
    for idx, pred in enumerate(predictions):
        entailment_score = float(pred[1]) if len(pred) > 1 else float(pred[0])
        
        chunk_info = {
            "id": retrieved_ids[idx],
            "text": retrieved_texts[idx],
            "metadata": retrieved_metadatas[idx],
            "entailment_score": entailment_score
        }
        
        if entailment_score >= ENTAILMENT_THRESHOLD:
            verified_chunks.append(chunk_info)
            print(f"    [+] ACCEPTED [{chunk_info['id']}]: Score = {entailment_score:.4f}")
        else:
            print(f"    [-] REJECTED [{chunk_info['id']}]: Score = {entailment_score:.4f} (Below threshold)")
            
    print(f"\n[*] Filtered down to {len(verified_chunks)} verified fact-bearing chunks for the SLM generator.")
    return verified_chunks

if __name__ == "__main__":
    embed_model, ce_model, collection = initialize_gate()
    sample_query = "What is the optimal dissolved oxygen level for Litopenaeus vannamei?"
    verified_context = evaluate_and_filter_chunks(sample_query, embed_model, ce_model, collection)