import os
import re
import torch
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
NLI_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "deberta_aquaculture_router")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "aqua_assist"

TOP_K = 5

# --- DECISION THRESHOLDS ---
ENTAILMENT_THRESHOLD = 0.75
NEUTRAL_THRESHOLD = 0.40
CONTRADICTION_THRESHOLD = 0.30

# --- HARDWARE ACCELERATION SELECTOR ---
# Automatically detects Apple Silicon (MPS), NVIDIA CUDA, or falls back to CPU
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Retrieval Quality Filter (Lower distance = better match. 0.80 is a strict cutoff for BGE-M3)
MAX_DISTANCE_THRESHOLD = 0.80

def detect_conflicting_measurements(farmer_query):
    """
    Precise conflict detector.
    Only triggers if multiple unique values exist for the *same* parameter.
    """
    text = farmer_query.lower()
    
    # Extract numbers near specific parameter labels
    do_pattern = r"(?:do|dissolved oxygen|oxygen)[^\d]{0,25}(\d+(?:\.\d+)?)"
    ph_pattern = r"\bph[^\d]{0,25}(\d+(?:\.\d+)?)"
    
    do_matches = re.findall(do_pattern, text)
    ph_matches = re.findall(ph_pattern, text)
    
    do_values = set(float(val) for val in do_matches)
    ph_values = set(float(val) for val in ph_matches)
    
    conflicts = []
    if len(do_values) > 1:
        conflicts.append(f"Dissolved Oxygen readings: {do_values}")
    if len(ph_values) > 1:
        conflicts.append(f"pH readings: {ph_values}")
        
    if conflicts:
        print(f"\n[!] CONFLICT GATE TRIGGERED: Contradictory measurements found for {', '.join(conflicts)}")
        return True
        
    return False

def load_engine():
    """Load BGE-M3, fine-tuned DeBERTa CrossEncoder, and ChromaDB."""
    print("[*] Loading BGE-M3...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

    print("[*] Loading fine-tuned DeBERTa CrossEncoder...")
    nli_model = CrossEncoder(NLI_MODEL_PATH, device=DEVICE)

    print("[*] Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    print("[+] Diagnostic engine loaded successfully.")
    return embedding_model, nli_model, collection

def retrieve_evidence(hypothesis, embedding_model, collection, top_k=5):
    """
    Phase 3 Retrieval Quality Gate: Enforces strict BGE-M3 distance thresholds 
    and filters out low-content micro-chunks.
    """
    embedding = embedding_model.encode(
        [hypothesis],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    evidence = []

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        text = results["documents"][0][i]
        
        # 1. Strict Distance Cutoff
        if distance > MAX_DISTANCE_THRESHOLD:  
            print(f"[!] Dropping low-relevance chunk {results['ids'][0][i]} (Distance: {distance:.4f} > {MAX_DISTANCE_THRESHOLD} threshold)")
            continue

        # 2. Minimum Content Length Filter
        if len(text.strip()) < 20:
            print(f"[!] Dropping micro-chunk {results['ids'][0][i]} (Too short for valid evidence)")
            continue

        evidence.append({
            "id": results["ids"][0][i],
            "text": text,
            "metadata": results["metadatas"][0][i],
            "distance": distance
        })

    return sorted(evidence, key=lambda x: x["distance"])[:top_k]

def evaluate_evidence(hypothesis, evidence_chunks, nli_model):
    """Evaluates NLI entailment, contradiction, and neutral probabilities using DeBERTa CrossEncoder."""
    evaluated = []
    MAX_NLI_WORDS = 50
    pairs = []

    for chunk in evidence_chunks:
        words = chunk["text"].split()
        nli_text = " ".join(words[:MAX_NLI_WORDS])
        pairs.append([nli_text, hypothesis])

    logits = nli_model.predict(pairs)

    for chunk, logit in zip(evidence_chunks, logits):
        probabilities = torch.softmax(
            torch.tensor(logit),
            dim=0
        ).tolist()

        result = {
            **chunk,
            "contradiction": float(probabilities[0]),
            "entailment": float(probabilities[1]),
            "neutral": float(probabilities[2])
        }
        evaluated.append(result)

    return evaluated

def make_decision(evaluated_chunks):
    """
    Hybrid Routing Logic: Uses BGE-M3 distance for confident symptom-matching, 
    and DeBERTa NLI as a safety net for ambiguous queries.
    """
    if not evaluated_chunks:
        return "CLARIFY"

    # 1. THE CONFIDENCE BYPASS (The Fix)
    # If BGE-M3 finds an exceptionally strong semantic match (distance < 0.40),
    # it means the database explicitly recognizes this severe symptom. 
    # Bypass the NLI contradiction paradox and provide the emergency remedy.
    high_confidence_matches = [
        chunk for chunk in evaluated_chunks 
        if chunk.get("distance", 1.0) < 0.40
    ]
    if high_confidence_matches:
        return "ANSWER"

    # 2. HARD CONTRADICTION (The Safety Net)
    # If the distance is weak but the statement is highly contradictory 
    # (e.g., pouring bleach), trigger the safety abort.
    hard_contradictions = [
        chunk for chunk in evaluated_chunks
        if chunk["contradiction"] >= 0.90 
    ]
    if hard_contradictions:
        return "ABSTAIN"

    # 3. STANDARD ENTAILMENT
    valid_answers = [
        chunk for chunk in evaluated_chunks
        if chunk["entailment"] >= 0.75
    ]
    if valid_answers:
        return "ANSWER"

    return "CLARIFY"

def run_diagnostic(hypothesis, embedding_model, nli_model, collection):
    """Run retrieval + NLI verification + three-way routing."""
    print(f"\n[?] Hypothesis: {hypothesis}")
    print("\n[*] Retrieving evidence...")

    evidence = retrieve_evidence(hypothesis, embedding_model, collection)

    if not evidence:
        print("[-] No highly relevant evidence found in the database. Routing to CLARIFY.")
        return {
            "hypothesis": hypothesis,
            "decision": "CLARIFY",
            "evidence": []
        }

    print(f"[+] Retrieved {len(evidence)} highly relevant evidence chunks.")
    print("\n[*] Running NLI verification...")

    evaluated = evaluate_evidence(hypothesis, evidence, nli_model)
    decision = make_decision(evaluated)
    
    print(f"\n[DECISION] {decision}")

    return {
        "hypothesis": hypothesis,
        "decision": decision,
        "evidence": evaluated
    }