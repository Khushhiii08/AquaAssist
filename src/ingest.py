import os
import glob
import re
import fitz  # PyMuPDF
import spacy
import ollama
import torch
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
OLLAMA_MODEL = "qwen2.5:1.5b"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SRC_MODEL = "facebook/nllb-200-distilled-600M"
CHROMA_BATCH_SIZE = 1000  # Safe incremental save limit

print(f"[*] Initializing hardware acceleration: {DEVICE.upper()}")

# --- LOAD MODELS ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[!] spaCy model not found. Run: python -m spacy download en_core_web_sm")
    exit()

print("[*] Loading NLLB-200 in FP16...")
tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(SRC_MODEL, torch_dtype=torch.float16).to(DEVICE)
translator_model.eval()

# --- TASK 1.1: SANITIZATION & EXTRACTION ---
def sanitize_pdf_text(raw_text):
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw_text)
    text = re.sub(r'(?<![.?!])\n+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def load_and_sanitize_pdfs(pdf_directory="data/raw_pdfs"):
    print(f"[*] Scanning '{pdf_directory}' for PDFs...")
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    corpus_text = ""
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                corpus_text += sanitize_pdf_text(page.get_text("text")) + "\n\n"
        except Exception as e:
            pass
    return corpus_text

# --- TASK 1.2: SEMANTIC TRIPLET CHUNKING ---
def extract_semantic_triplets(corpus_text):
    print("[*] Parsing semantic triplets (Subject-Predicate-Object)...")
    paragraphs = corpus_text.split("\n\n")
    structured_chunks = []
    
    for para in tqdm(paragraphs, desc="Parsing"):
        if len(para.strip()) < 20: continue
        doc = nlp(para)
        for sent in doc.sents:
            subject, verb, obj = [], [], []
            for token in sent:
                if "subj" in token.dep_: subject.append(token.text)
                elif "ROOT" in token.dep_ or token.pos_ == "VERB": verb.append(token.text)
                elif "obj" in token.dep_ or "attr" in token.dep_: obj.append(token.text)
            
            if subject and verb and obj:
                actionable_chunk = f"Observation: {' '.join(subject)} {' '.join(verb)} {' '.join(obj)}. Context: {sent.text.strip()}"
                structured_chunks.append(actionable_chunk)
    return structured_chunks

# --- TASK 1.3: LLM SYNTHESIS & NOISE FILTERING ---
def synthesize_advice(triplet_chunk):
    system_prompt = "You are an expert aquaculture assistant. Be concise, empathetic, and strictly factual."
    user_prompt = f"Read this observation extracted from a scientific manual. If it contains actionable advice, biological facts, or environmental warnings for a farmer, rewrite it into a short, simple, 1-2 sentence response. Use a helpful tone.\nIf it is just administrative, statistical, or academic noise (like software names or methodology), output exactly and only the word: SKIP\nData:\n{triplet_chunk}"
    
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        return response['message']['content'].strip()
    except Exception:
        return "SKIP"

# --- TASK 1.4: NMT TRANSLATION WITH LEXICON ENFORCER ---
def translate_to_telugu(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("tel_Telu")
    
    with torch.no_grad():
        translated_tokens = translator_model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=256, num_beams=1)
        
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].strip()
    
    # Domain Lexicon Enforcer: Correct known 600M edge-model artifacts
    translation = translation.replace("గ్రెయిన్", "రొయ్యలు") # shrimp
    translation = translation.replace("గ్రెడ్లు", "రొయ్యలు")   # shrimp
    translation = translation.replace("కుట్టడం", "గడ్డకట్టడం") # clotting
    return translation

# --- TASK 1.5: DATABASE COMPILATION ---
def build_database(chunks, embedding_function):
    db_path = os.path.join("./data", "chroma_db")
    print(f"[*] Initializing ChromaDB at {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="aqua_assist", embedding_function=embedding_function)

    documents_buffer, metadatas_buffer, ids_buffer = [], [], []
    global_idx = 0
    
    print("[*] Beginning LLM Synthesis and Translation Pipeline...")
    for chunk in tqdm(chunks, desc="Processing Knowledge Base"):
        # 1. Synthesize and filter
        english_advice = synthesize_advice(chunk)
        if english_advice == "SKIP" or "SKIP" in english_advice:
            continue
            
        # 2. Translate the cleaned advice
        telugu_advice = translate_to_telugu(english_advice)
        
        # 3. Buffer for ChromaDB
        documents_buffer.append(chunk)  # We embed the original triplet for strict retrieval
        ids_buffer.append(f"prop_chunk_{global_idx}")
        metadatas_buffer.append({
            "category": "Aquaculture Advisory",
            "english_synthesis": english_advice,
            "telugu_translation": telugu_advice
        })
        global_idx += 1
        
        # 4. Safe incremental saving
        if len(documents_buffer) >= CHROMA_BATCH_SIZE:
            collection.upsert(documents=documents_buffer, metadatas=metadatas_buffer, ids=ids_buffer)
            documents_buffer, metadatas_buffer, ids_buffer = [], [], []
            if DEVICE == "mps": torch.mps.empty_cache()
            
    if documents_buffer:
        collection.upsert(documents=documents_buffer, metadatas=metadatas_buffer, ids=ids_buffer)
        
    print(f"[*] Database rebuild complete! Saved {global_idx} fully synthesized conversational items.")

if __name__ == "__main__":
    raw_corpus = load_and_sanitize_pdfs("data/raw_pdfs")
    if raw_corpus:
        triplet_chunks = extract_semantic_triplets(raw_corpus)
        
        print("[*] Loading BAAI/bge-small-en-v1.5 embedding model...")
        bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
        
        build_database(triplet_chunks, embedding_function=bge_ef)