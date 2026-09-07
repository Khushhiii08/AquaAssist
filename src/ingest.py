import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"

import json
import re
from pathlib import Path
import chromadb
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(4)

RAW_PDFS_DIR = "./data/raw_pdfs"
PROCESSED_DATA_DIR = "./data/processed"
CHROMA_DB_DIR = "./data/chroma_db"
CHUNKS_PATH = os.path.join(PROCESSED_DATA_DIR, "chunks.json")

def extract_metadata_and_chunk(text, source_filename):
    chunks = []
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    
    if not raw_paragraphs:
        raw_paragraphs = [s.strip() for s in text.split(". ") if len(s.strip()) > 50]

    for para in raw_paragraphs:
        lower_para = para.lower()
        
        if any(w in lower_para for w in ["wssv", "ehp", "vibrio", "disease", "pathogen", "mortality", "syndrome"]):
            category = "Disease Diagnostics"
        elif any(w in lower_para for w in ["oxygen", "ph", "salinity", "ammonia", "tan", "alkalinity", "temperature", "ppm", "ppt"]):
            category = "Water Quality Limits"
        elif any(w in lower_para for w in ["dosage", "treatment", "probiotic", "application", "gram", "ml", "disinfection"]):
            category = "Dosage & Treatment"
        else:
            category = "Husbandry & Management"

        if "vannamei" in lower_para:
            species = "Litopenaeus vannamei"
        elif "rohu" in lower_para or "labeo rohita" in lower_para:
            species = "Labeo rohita"
        else:
            species = "General Aquaculture"

        parameters = {}
        ph_match = re.search(r"ph\s*([0-9\.-]+)", lower_para)
        if ph_match:
            parameters["pH"] = ph_match.group(1)
            
        do_match = re.search(r"([0-9\.]+)\s*(?:mg/l|ppm)", lower_para)
        if do_match:
            parameters["Dissolved_Oxygen"] = do_match.group(1)

        chunks.append({
            "source_document": source_filename,
            "domain_category": category,
            "target_species": species,
            "extracted_parameters": parameters,
            "text": para
        })

    return chunks

def run_hybrid_ingestion():
    print("[*] Initializing Phase 1: Hybrid Corpus Ingestion & Indexing...")
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    
    all_chunks = []
    
    if not os.path.exists(RAW_PDFS_DIR) or not os.listdir(RAW_PDFS_DIR):
        print(f"[!] Warning: No PDFs found in {RAW_PDFS_DIR}. Creating placeholder corpus.")
        os.makedirs(RAW_PDFS_DIR, exist_ok=True)
        sample_path = os.path.join(RAW_PDFS_DIR, "ICAR-CIBA_Sample_Guidelines.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("ICAR-CIBA guidelines for Litopenaeus vannamei culture. Optimal dissolved oxygen must be maintained above 4.0 mg/L. pH range should be 7.5 to 8.5. For White Spot Syndrome Virus (WSSV), immediate quarantine and strict biosecurity protocols are required. Total Ammonia Nitrogen (TAN) must not exceed 0.05 mg/L.")
            
    global_chunk_counter = 0

    for filename in os.listdir(RAW_PDFS_DIR):
        file_path = os.path.join(RAW_PDFS_DIR, filename)
        file_chunks = []
        
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            file_chunks = extract_metadata_and_chunk(content, filename)
        elif filename.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                content = "".join([page.extract_text() or "" for page in reader.pages])
                file_chunks = extract_metadata_and_chunk(content, filename)
            except Exception as e:
                print(f"[-] Error reading PDF {filename}: {e}")

        # Assign strictly unique global IDs across all files
        for chunk in file_chunks:
            chunk["chunk_id"] = f"CHK-{global_chunk_counter:05d}"
            global_chunk_counter += 1
            all_chunks.append(chunk)

    if not all_chunks:
        raise ValueError("[-] No text chunks generated.")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4)
    print(f"[+] Successfully saved {len(all_chunks)} structured metadata chunks to {CHUNKS_PATH}")

    print("[*] Loading BAAI/bge-m3 model with Apple Silicon MPS acceleration...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] Target Hardware Device: {device}")
    
    embed_model = SentenceTransformer("BAAI/bge-m3", device=device)

    print(f"[*] Connecting to local ChromaDB at {CHROMA_DB_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    collection_name = "aquaculture_knowledge"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
        
    collection = client.create_collection(name=collection_name)

    print("[*] Encoding chunks in safe batches and inserting into ChromaDB...")
    ids = [c["chunk_id"] for c in all_chunks]
    texts = [c["text"] for c in all_chunks]
    metadatas = [{
        "source": c["source_document"],
        "category": c["domain_category"],
        "species": c["target_species"],
        "parameters": json.dumps(c["extracted_parameters"])
    } for c in all_chunks]

    embeddings = embed_model.encode(
        texts, 
        batch_size=8, 
        show_progress_bar=True,
        device=device
    ).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"[+] Phase 1 Complete! Indexed {len(ids)} hybrid-backed chunks into ChromaDB collection '{collection_name}'.")

if __name__ == "__main__":
    run_hybrid_ingestion()