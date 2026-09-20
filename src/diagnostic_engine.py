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

ENTAILMENT_THRESHOLD = 0.75
NEUTRAL_THRESHOLD = 0.40
CONTRADICTION_THRESHOLD = 0.30


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


def retrieve_evidence(hypothesis, embedding_model, collection, top_k=TOP_K):
    """Retrieve relevant evidence chunks from ChromaDB."""

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
        evidence.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return evidence


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
    Convert multiple NLI results into one three-way routing decision.
    """

    if not evaluated_chunks:
        return "CLARIFY"

    # ---------------------------------------------------------
    # 1. Identify strong contradiction evidence
    # ---------------------------------------------------------

    strong_contradictions = [
        chunk
        for chunk in evaluated_chunks
        if (
            chunk["contradiction"] > CONTRADICTION_THRESHOLD
            and chunk["contradiction"] > chunk["entailment"]
            and chunk["contradiction"] > chunk["neutral"]
        )
    ]

    highly_contradictory = [
        chunk
        for chunk in strong_contradictions
        if chunk["contradiction"] >= 0.75
    ]

    if len(highly_contradictory) >= 2:
        return "ABSTAIN"

    # ---------------------------------------------------------
    # 2. Identify strong supporting evidence
    # ---------------------------------------------------------

    strong_entailments = [
        chunk
        for chunk in evaluated_chunks
        if chunk["entailment"] >= ENTAILMENT_THRESHOLD
    ]

    if strong_entailments:
        return "ANSWER"

    # ---------------------------------------------------------
    # 3. Evidence is insufficient
    # ---------------------------------------------------------

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

    print(f"[+] Retrieved {len(evidence)} evidence chunks.")

    print("\n[*] Running NLI verification...")

    evaluated = evaluate_evidence(
        hypothesis,
        evidence,
        nli_model
    )

    for i, chunk in enumerate(evaluated, 1):
        print(f"\nEvidence {i}: {chunk['id']}")
        print(f"  Distance:      {chunk['distance']:.4f}")
        print(f"  Topic:         {chunk['metadata'].get('topic')}")
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