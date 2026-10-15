import os
import glob
import re
import json
import pymupdf as fitz
import spacy
from tqdm import tqdm
import torch
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from llama_cpp import Llama, LlamaGrammar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen-2.5-1.5b-instruct.gguf")
GRAMMAR_PATH = os.path.join(PROJECT_ROOT, "constraints.gbnf")
SRC_MODEL = "facebook/nllb-200-distilled-600M"
CHROMA_BATCH_SIZE = 100

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing PyTorch hardware acceleration: {DEVICE.upper()}")

# --- LOAD SPACY ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[!] spaCy model not found. Run: python -m spacy download en_core_web_sm")
    exit(1)

# --- LOAD NLLB-200 ---
print("[*] Loading NLLB-200...")
tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)
model_dtype = torch.float16 if DEVICE != "cpu" else torch.float32
translator_model = AutoModelForSeq2SeqLM.from_pretrained(
    SRC_MODEL, 
    torch_dtype=model_dtype
).to(DEVICE)
translator_model.eval()

# --- LOAD QWEN VIA LLAMA.CPP (CPU Safe for Ingestion) ---
print(f"[*] Loading Qwen 1.5B via llama.cpp from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model GGUF not found at: {MODEL_PATH}")

# Explicitly use n_gpu_layers=0 during batch ingestion to prevent Metal resource clashes with PyTorch MPS
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=False
)

strict_json_grammar = None
if os.path.exists(GRAMMAR_PATH):
    try:
        with open(GRAMMAR_PATH, "r", encoding="utf-8") as f:
            strict_json_grammar = LlamaGrammar.from_string(f.read())
        print("[+] Loaded strict GBNF grammar constraints.")
    except Exception as ge:
        print(f"[!] Warning: Failed parsing GBNF ({ge}). Continuing without grammar.")
else:
    print(f"[!] Notice: No grammar file at {GRAMMAR_PATH}. Running unconstrained JSON.")


def classify_chunk_metadata(text: str, filename: str) -> dict:
    """Categorizes chunk into discrete facets for ChromaDB filtering."""
    t_low = (text + " " + filename).lower()

    # Species classification
    shrimp_markers = {"shrimp", "vannamei", "monodon", "prawn", "pleopod", "rostrum", "ehp", "wsv"}
    fish_markers = {"fish", "carp", "tilapia", "rohu", "catla", "mrigal", "finfish", "fingerling"}

    has_shrimp = bool(set(re.findall(r'\b\w+\b', t_low)) & shrimp_markers)
    has_fish = bool(set(re.findall(r'\b\w+\b', t_low)) & fish_markers)

    if has_shrimp and not has_fish:
        species = "shrimp"
    elif has_fish and not has_shrimp:
        species = "fish"
    else:
        species = "general"

    # Habitat classification
    if any(k in t_low for k in ["freshwater", "fresh water", "inland"]):
        habitat = "freshwater"
    elif any(k in t_low for k in ["brackish", "saline", "marine", "estuary"]):
        habitat = "brackish"
    else:
        habitat = "general"

    # Primary Limnological Topic
    topic = "general_management"
    if any(k in t_low for k in ["dissolved oxygen", "hypoxia", "aeration"]):
        topic = "dissolved_oxygen"
    elif any(k in t_low for k in ["ammonia", "tan", "nitrite", "toxic gas"]):
        topic = "ammonia_nitrogen"
    elif any(k in t_low for k in ["alkalinity", "calcium carbonate", "hardness"]):
        topic = "alkalinity"
    elif any(k in t_low for k in ["ph", "acidity"]):
        topic = "pH"
    elif any(k in t_low for k in ["temperature", "thermal"]):
        topic = "temperature"
    elif any(k in t_low for k in ["disease", "pathology", "mortality", "lesion"]):
        topic = "pathology"

    return {
        "species": species,
        "habitat": habitat,
        "primary_topic": topic
    }


def load_and_sanitize_pdfs(pdf_directory="data/raw_pdfs"):
    resolved_pdf_dir = os.path.join(PROJECT_ROOT, pdf_directory) if not os.path.isabs(pdf_directory) else pdf_directory
    print(f"[*] Scanning '{resolved_pdf_dir}' for PDFs...")
    pdf_files = glob.glob(os.path.join(resolved_pdf_dir, "*.pdf"))
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


def extract_semantic_triplets(semantic_chunks):
    print("[*] Parsing semantic triplets (Subject-Predicate-Object)...")
    structured_chunks = []
    bad_keywords = ["et al", "figure", "table ", "references", "kg/ha", "statistically", "spss", "methodology"]

    for chunk in tqdm(semantic_chunks, desc="spaCy Parsing"):
        para = chunk["text"]

        if any(bad in para.lower() for bad in bad_keywords):
            continue

        doc = nlp(para)
        triplets_in_para = []

        for sent in doc.sents:
            subject, verb, obj = [], [], []
            for token in sent:
                if "subj" in token.dep_:
                    subject.append(token.text)
                elif "ROOT" in token.dep_ or token.pos_ == "VERB":
                    verb.append(token.text)
                elif "obj" in token.dep_ or "attr" in token.dep_:
                    obj.append(token.text)

            if subject and verb and obj and (len(subject) + len(verb) + len(obj) >= 4):
                triplets_in_para.append(f"({' '.join(subject)} -> {' '.join(verb)} -> {' '.join(obj)})")

        if triplets_in_para:
            actionable_chunk = f"Source Text: {para}\nKey Entities: {', '.join(triplets_in_para)}"
            structured_chunks.append({
                "text": actionable_chunk,
                "metadata": chunk["metadata"]
            })

    return structured_chunks


def synthesize_advice(chunk_text: str) -> dict:
    user_prompt = f"Extract actionable aquaculture advice from this passage:\n{chunk_text}"
    try:
        kwargs = {
            "messages": [
                {"role": "system", "content": "Extract water parameters, symptoms, and actions. Output JSON."},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.1
        }
        if strict_json_grammar:
            kwargs["grammar"] = strict_json_grammar

        response = llm.create_chat_completion(**kwargs)
        payload = json.loads(response['choices'][0]['message']['content'])
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def translate_to_telugu(text: str) -> str:
    if not text or len(text.strip()) < 5:
        return ""
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    forced_bos = tokenizer.convert_tokens_to_ids("tel_Telu")
    with torch.no_grad():
        tokens = translator_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=256
        )
    out = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    return (
        out.replace("గ్రెయిన్", "రొయ్యలు")
           .replace("గ్రెడ్లు", "రొయ్యలు")
           .replace("కుట్టడం", "గడ్డకట్టడం")
    )


def build_database(chunks, embedding_function):
    db_path = os.path.join(PROJECT_ROOT, "data", "chroma_db")
    print(f"[*] Connecting to ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection("aqua_assist")
    except Exception:
        pass

    collection = client.create_collection(name="aqua_assist", embedding_function=embedding_function)

    docs_buf, metas_buf, ids_buf = [], [], []
    idx = 0

    for chunk in tqdm(chunks, desc="Ingesting Manuals"):
        meta_tags = classify_chunk_metadata(chunk["text"], chunk["metadata"]["source"])
        synth = synthesize_advice(chunk["text"])

        action_en = synth.get("recommended_action_english") or synth.get("recommended_action", "")
        if not action_en and "english_synthesis" in synth:
            action_en = synth["english_synthesis"]

        action_te = translate_to_telugu(action_en) if action_en else ""

        docs_buf.append(chunk["text"])
        ids_buf.append(f"chunk_{idx}")
        metas_buf.append({
            "source": chunk["metadata"]["source"],
            "page": int(chunk["metadata"]["page"]),
            "species": meta_tags["species"],
            "habitat": meta_tags["habitat"],
            "primary_topic": meta_tags["primary_topic"],
            "english_synthesis": action_en,
            "telugu_translation": action_te
        })
        idx += 1

        if len(docs_buf) >= CHROMA_BATCH_SIZE:
            collection.upsert(documents=docs_buf, metadatas=metas_buf, ids=ids_buf)
            docs_buf, metas_buf, ids_buf = [], [], []

    if docs_buf:
        collection.upsert(documents=docs_buf, metadatas=metas_buf, ids=ids_buf)

    print(f"[+] Ingestion complete. Indexed {idx} chunks with metadata facets.")


if __name__ == "__main__":
    raw_chunks = load_and_sanitize_pdfs("data/raw_pdfs")
    if raw_chunks:
        triplet_chunks = extract_semantic_triplets(raw_chunks)
        bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3", 
            device=DEVICE
        )
        build_database(triplet_chunks, embedding_function=bge_ef)
    else:
        print("[!] No PDF documents found in data/raw_pdfs/")