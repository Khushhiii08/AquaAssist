import re
import os
import chromadb
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder


# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
NLI_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "deberta_aquaculture_router")

EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "aquaculture_knowledge"

TOP_K = 5

# --- DECISION THRESHOLDS ---
ENTAILMENT_THRESHOLD = 0.75
NEUTRAL_THRESHOLD = 0.40
CONTRADICTION_THRESHOLD = 0.30

# NEW: Retrieval Quality Filter (Lower distance = better match. 0.80 is a strict cutoff for BGE-M3)
MAX_DISTANCE_THRESHOLD = 0.80

def detect_conflicting_measurements(farmer_query):
    """
    ISSUE-05: Precise conflict detector.
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
    # Only a conflict if there are 2+ distinct numbers for DO or 2+ distinct numbers for pH
    if len(do_values) > 1:
        conflicts.append(f"Dissolved Oxygen readings: {do_values}")
    if len(ph_values) > 1:
        conflicts.append(f"pH readings: {ph_values}")
        
    if conflicts:
        print(f"\n[!] CONFLICT GATE TRIGGERED: Contradictory measurements found for {', '.join(conflicts)}")
        return True
        
    return False

def load_engine():
    """Load BGE-M3, fine-tuned DeBERTa, and ChromaDB."""
    print("[*] Loading BGE-M3...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("[*] Loading fine-tuned DeBERTa...")
    nli_model = CrossEncoder(NLI_MODEL_PATH)

    print("[*] Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    print("[+] Diagnostic engine loaded successfully.")

    return embedding_model, nli_model, collection


def retrieve_evidence(hypothesis, embedding_model, collection, top_k=10):
    """Retrieve relevant evidence chunks from ChromaDB using similarity ranking instead of arbitrary distance dropoffs."""
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
        
        # BGE-M3 distance sanity check (safely capture anything reasonably close)
        if distance > 1.2:  
            print(f"[!] Dropping completely unrelated chunk {results['ids'][0][i]} (Distance: {distance:.4f})")
            continue

        evidence.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": distance
        })

    # Sort by closest distance first to guarantee the strongest chunk is evaluated by NLI
    evidence = sorted(evidence, key=lambda x: x["distance"])
    return evidence[:5]  # Keep top 5 best matches


def evaluate_evidence(hypothesis, evidence_chunks, nli_model):
    evaluated = []

    print("\n[DEBUG] NLI INPUTS")
    for i, chunk in enumerate(evidence_chunks, 1):
        print(f"\n--- Evidence {i}: {chunk['id']} ---")
        print("TEXT:", chunk["text"][:500])
        print("HYPOTHESIS:", hypothesis)

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
    """Convert multiple NLI results into one three-way routing decision."""
    if not evaluated_chunks:
        return "CLARIFY"

    # 1. Check for severe multi-chunk contradiction (ABSTAIN)
    strong_contradictions = [
        chunk for chunk in evaluated_chunks
        if chunk["contradiction"] > 0.75 and chunk["contradiction"] > chunk["entailment"]
    ]
    if len(strong_contradictions) >= 2:
        return "ABSTAIN"

    # 2. Check for strong supporting evidence (ANSWER)
    valid_answers = [
        chunk for chunk in evaluated_chunks
        if chunk["entailment"] >= ENTAILMENT_THRESHOLD and chunk["contradiction"] < 0.30
    ]

    if valid_answers:
        return "ANSWER"

    return "CLARIFY"


def run_diagnostic(hypothesis, embedding_model, nli_model, collection):
    """Run retrieval + NLI verification + three-way routing."""
    print(f"\n[?] Hypothesis: {hypothesis}")
    print("\n[*] Retrieving evidence...")

    evidence = retrieve_evidence(
        hypothesis,
        embedding_model,
        collection
    )

    # SHORT-CIRCUIT: If ChromaDB only returned garbage, skip NLI and instantly clarify
    if not evidence:
        print("[-] No highly relevant evidence found in the database. Routing to CLARIFY.")
        decision = "CLARIFY"
        print(f"\n[DECISION] {decision}")
        return {
            "hypothesis": hypothesis,
            "decision": decision,
            "evidence": []
        }

    print(f"[+] Retrieved {len(evidence)} highly relevant evidence chunks.")
    print("\n[*] Running NLI verification...")

    evaluated = evaluate_evidence(
        hypothesis,
        evidence,
        nli_model
    )

    for i, chunk in enumerate(evaluated, 1):
        print(f"\nEvidence {i}: {chunk['id']}")
        print(f"  Distance:      {chunk['distance']:.4f}")
        print(f"  Topic:         {chunk['metadata'].get('category')}")
        print(f"  Species:       {chunk['metadata'].get('species')}")
        print(f"  Contradiction: {chunk['contradiction']:.4f}")
        print(f"  Entailment:    {chunk['entailment']:.4f}")
        print(f"  Neutral:       {chunk['neutral']:.4f}")

    decision = make_decision(evaluated)
    print(f"\n[DECISION] {decision}")

    return {
        "hypothesis": hypothesis,
        "decision": decision,
        "evidence": evaluated
    }


if __name__ == "__main__":
    from intent_extractor import extract_hypothesis

    farmer_query = (
        "My shrimp are not eating and the pond water became dark green."
    )

    print("\n[*] Extracting diagnostic hypothesis with Qwen-0.5B...")
    hypothesis = extract_hypothesis(farmer_query)
    print(f"[+] Hypothesis: {hypothesis}")

    embedding_model, nli_model, collection = load_engine()
    
    result = run_diagnostic(
        hypothesis,
        embedding_model,
        nli_model,
        collection
    )
    
    print(f"\nFinal decision: {result['decision']}")