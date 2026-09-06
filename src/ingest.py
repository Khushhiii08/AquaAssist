import pymupdf
import json
import re
import os
import glob

# Dynamically resolve project root: /aquassist/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Configuration anchored to project root
CHUNK_SIZE = 150
OVERLAP = 30
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "chunks.json")

def clean_text(text):
    """Removes excessive spaces, newlines, and weird PDF formatting."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tag_metadata(text):
    """Rule-based tagger for aquaculture domain invariants."""
    text_lower = text.lower()
    metadata = {
        "species": "general",
        "topic": "general"
    }
    
    # Species tagging
    if any(word in text_lower for word in ["shrimp", "vannamei", "prawn", "monodon", "penaeus"]):
        metadata["species"] = "shrimp"
    elif any(word in text_lower for word in ["fish", "rohu", "catla", "carp"]):
        metadata["species"] = "fish"
        
    # Topic tagging
    if any(word in text_lower for word in ["ph", "ammonia", "oxygen", "salinity", "temperature", "mg/l", "ppm", "alkalinity"]):
        metadata["topic"] = "water_quality"
    elif any(word in text_lower for word in ["disease", "virus", "white spot", "bacteria", "mortality", "treatment", "ehp", "spore", "pathology"]):
        metadata["topic"] = "disease_management"
    elif any(word in text_lower for word in ["feed", "protein", "pellet", "nutrition"]):
        metadata["topic"] = "nutrition"
        
    return metadata

def process_pdfs():
    print(f"[*] Project root identified at: {PROJECT_ROOT}")
    print(f"[*] Scanning for PDFs in: {PDF_DIR}")
    
    # Verify folder exists
    if not os.path.exists(PDF_DIR):
        print(f"[-] Directory does not exist: {PDF_DIR}")
        return

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"[-] No PDFs found in {PDF_DIR}. Ensure dummy_test.pdf is placed here.")
        return

    all_chunks = []
    global_chunk_id = 0

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"[+] Processing {file_name}...")
        
        try:
            doc = pymupdf.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + " "
                
            clean_full_text = clean_text(full_text)
            words = clean_full_text.split()
            print(f"    -> Extracted {len(words)} words from {file_name}")
            
            # Sliding window for overlapping chunks
            for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
                chunk_words = words[i:i + CHUNK_SIZE]
                if len(chunk_words) < 40:
                    continue
                    
                chunk_text = " ".join(chunk_words)
                metadata = tag_metadata(chunk_text)
                metadata["source"] = file_name
                
                all_chunks.append({
                    "id": f"CHK-{global_chunk_id:05d}",
                    "text": chunk_text,
                    "metadata": metadata
                })
                global_chunk_id += 1
                
        except Exception as e:
            print(f"[-] Failed to read {file_name}: {e}")

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4)
        
    print(f"\n[*] SUCCESS: Extracted {len(all_chunks)} total chunks.")
    print(f"[*] Saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    process_pdfs()