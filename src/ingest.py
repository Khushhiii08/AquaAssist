import os
import glob
import re
import fitz  # PyMuPDF
import spacy
import json
import torch
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from llama_cpp import Llama, LlamaGrammar

# --- CONFIGURATION ---
MODEL_PATH = "models/qwen-2.5-1.5b-instruct.gguf"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SRC_MODEL = "facebook/nllb-200-distilled-600M"
CHROMA_BATCH_SIZE = 1000  

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

print("[*] Loading Qwen 1.5B via llama.cpp for Grammar Constrained Generation...")
llm = Llama(
    model_path=MODEL_PATH, 
    n_ctx=2048, 
    n_gpu_layers=-1,  # Forces 100% GPU offloading for M-series chips
    verbose=False
)
with open("constraints.gbnf", "r") as f:
    strict_json_grammar = LlamaGrammar.from_string(f.read())

# --- TASK 1.1: RECURSIVE CHUNKING ---
def load_and_sanitize_pdfs(pdf_directory="data/raw_pdfs"):
    print(f"[*] Scanning '{pdf_directory}' for PDFs...")
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    semantic_chunks = []
    
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                raw_text = doc[page_num].get_text("text")
                paragraphs = re.split(r'\n\s*\n', raw_text)
                
                for para in paragraphs:
                    cleaned_para = para.replace('\n', ' ').strip()
                    if len(cleaned_para) > 60:
                        semantic_chunks.append({
                            "text": cleaned_para,
                            "metadata": {
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1
                            }
                        })
        except Exception as e:
            print(f"[!] Error reading {pdf_path}: {e}")
            
    return semantic_chunks

# --- TASK 1.2: SEMANTIC TRIPLET CHUNKING (Consolidated) ---
def extract_semantic_triplets(semantic_chunks):
    print("[*] Parsing semantic triplets (Subject-Predicate-Object)...")
    structured_chunks = []
    bad_keywords = ["et al", "figure", "table ", "references", "kg/ha", "statistically", "spss", "methodology"]
    
    for chunk in tqdm(semantic_chunks, desc="spaCy Parsing"):
        para = chunk["text"]
        
        # Pre-filter out blatant academic noise
        if any(bad in para.lower() for bad in bad_keywords):
            continue

        doc = nlp(para)
        triplets_in_para = []
        
        for sent in doc.sents:
            subject, verb, obj = [], [], []
            for token in sent:
                if "subj" in token.dep_: subject.append(token.text)
                elif "ROOT" in token.dep_ or token.pos_ == "VERB": verb.append(token.text)
                elif "obj" in token.dep_ or "attr" in token.dep_: obj.append(token.text)
            
            if subject and verb and obj and (len(subject) + len(verb) + len(obj) >= 4):
                triplets_in_para.append(f"({' '.join(subject)} -> {' '.join(verb)} -> {' '.join(obj)})")
                
        # Only pass the chunk to the LLM if we actually found biological relationships
        if triplets_in_para:
            actionable_chunk = f"Source Text: {para}\nKey Entities: {', '.join(triplets_in_para)}"
            structured_chunks.append({
                "text": actionable_chunk,
                "metadata": chunk["metadata"]
            })
                
    return structured_chunks

# --- TASK 1.3: LLM SYNTHESIS (Crash-Proofed) ---
QWEN_SYSTEM_PROMPT = """
You are a pragmatic aquaculture diagnostician. 
STRICT CONSTRAINTS: 
1. IGNORE academic methodology, citations, and statistics. 
2. EXTRACT ONLY water quality parameters, disease symptoms, and actionable treatments.
3. BE CONCISE. Keep your recommended action under 2 sentences.
4. Output ONLY valid JSON.
"""

def synthesize_advice(chunk_text):
    user_prompt = f"Extract the actionable diagnostic advice from this data:\n{chunk_text}"
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            grammar=strict_json_grammar,
            max_tokens=1024, # Doubled buffer to ensure it always closes the JSON bracket
            temperature=0.1  # Low temperature stops the model from hallucinating loops
        )
        
        raw_json_str = response['choices'][0]['message']['content']
        return json.loads(raw_json_str)
        
    except json.JSONDecodeError:
        # We silently pass here so it doesn't clutter your terminal if a bad chunk slips through
        return None
    except Exception as e:
        print(f"\n[!] LLM Synthesis Error: {e}")
        return None

# --- TASK 1.4: NMT TRANSLATION WITH LEXICON ENFORCER ---
def translate_to_telugu(text):
    if not text:
        return ""
        
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("tel_Telu")
    
    with torch.no_grad():
        translated_tokens = translator_model.generate(
            **inputs, 
            forced_bos_token_id=forced_bos_token_id, 
            max_length=256, 
            num_beams=2,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
        
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].strip()
    translation = translation.replace("గ్రెయిన్", "రొయ్యలు").replace("గ్రెడ్లు", "రొయ్యలు").replace("కుట్టడం", "గడ్డకట్టడం")
    return translation

# --- TASK 1.5: DATABASE COMPILATION ---
def build_database(chunks, embedding_function):
    db_path = os.path.join("./data", "chroma_db")
    print(f"[*] Initializing ChromaDB at {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    
    # Wipe the old collection to prevent schema mismatch errors with new metadata
    try:
        client.delete_collection("aqua_assist")
    except Exception:
        pass
        
    collection = client.create_collection(name="aqua_assist", embedding_function=embedding_function)

    documents_buffer, metadatas_buffer, ids_buffer = [], [], []
    global_idx = 0
    
    print("[*] Beginning LLM Synthesis and Translation Pipeline...")
    for chunk in tqdm(chunks, desc="Building Knowledge Base"):
        
        synth_payload = synthesize_advice(chunk["text"])
        
        # Ensure payload exists and has values
        if not synth_payload or not isinstance(synth_payload, dict):
            continue
            
        english_action = synth_payload.get("recommended_action_telugu", "")
        if not english_action or "skip" in english_action.lower():
            continue
            
        telugu_advice = translate_to_telugu(english_action)
        
        documents_buffer.append(chunk["text"])
        ids_buffer.append(f"prop_chunk_{global_idx}")
        metadatas_buffer.append({
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "category": "Aquaculture Advisory",
            "english_synthesis": english_action,
            "telugu_translation": telugu_advice
        })
        global_idx += 1
        
        if len(documents_buffer) >= CHROMA_BATCH_SIZE:
            collection.upsert(documents=documents_buffer, metadatas=metadatas_buffer, ids=ids_buffer)
            documents_buffer, metadatas_buffer, ids_buffer = [], [], []
            
    if documents_buffer:
        collection.upsert(documents=documents_buffer, metadatas=metadatas_buffer, ids=ids_buffer)
        
    print(f"[*] Database rebuild complete! Saved {global_idx} fully synthesized items.")

if __name__ == "__main__":
    raw_chunks = load_and_sanitize_pdfs("data/raw_pdfs")
    
    if raw_chunks:
        # Pass through the restored spaCy logic
        triplet_chunks = extract_semantic_triplets(raw_chunks)
        
        print("[*] Loading BAAI/bge-m3 embedding model...")
        bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
        
        build_database(triplet_chunks, embedding_function=bge_ef)