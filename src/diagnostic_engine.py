"""AquaAssist: Hardened Diagnostic Engine & Grounded NLI Gate
Executes:
1. Multi-reading conflict and biological feasibility checks on telemetry.
2. Content-density species isolation during ChromaDB retrieval.
3. Declarative Premise-Hypothesis NLI evaluation (eliminates Neutral question bias).
4. Strictly calibrated decision routing (EMERGENCY, CONFLICT, ANSWER, CLARIFY, ABSTAIN).
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import torch
from src.sync_manager import get_active_db_path
import spacy

# --- INITIALIZE SPACY MODEL ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DB_PATH = get_active_db_path()
NLI_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "deberta_aquaculture_router"
)

COLLECTION_NAME = "aqua_assist"
TOP_K = 5

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

MAX_DISTANCE_THRESHOLD = 0.85
ENTAILMENT_THRESHOLD = 0.65
CONTRADICTION_THRESHOLD = 0.40
NEUTRAL_THRESHOLD = 0.35

BIOLOGICAL_LIMITS = {
    "DO": (0.0, 22.0),
    "pH": (3.5, 11.5),
    "temperature": (5.0, 45.0),
    "salinity": (0.0, 60.0),
    "TAN": (0.0, 20.0),
    "alkalinity": (10.0, 600.0),
}


def evaluate_telemetry_health_and_conflicts(
    session_state: Dict[str, Any],
) -> Tuple[bool, Optional[str], Optional[str]]:
  """Inspects structured telemetry for internal contradictions, biological bounds, and acute emergencies."""
  telemetry = session_state.get("telemetry", {})
  if not telemetry:
    return False, None, None

  # 1. Multi-Reading Conflict Detection (e.g. T11: one report 2 mg/L, another 8 mg/L)
  for param, records in telemetry.items():
    if isinstance(records, list) and len(records) > 1:
      values = [r["value"] for r in records if isinstance(r, dict) and r.get("value") is not None]
      if len(values) > 1:
        spread = max(values) - min(values)
        if param == "DO" and spread >= 2.0:
          return (
              True,
              "CONFLICT_READINGS",
              f"Contradictory Dissolved Oxygen readings reported: {values} mg/L. Re-calibration required.",
          )
        if param == "pH" and spread >= 1.5:
          return (
              True,
              "CONFLICT_READINGS",
              f"Contradictory pH readings reported: {values}. Re-calibration required.",
          )

  # 2. Biological Boundary Verification
  for param, records in telemetry.items():
    items = records if isinstance(records, list) else [records]
    for item in items:
      if not isinstance(item, dict):
        continue
      val = item.get("value")
      if val is None:
        continue
      if param in BIOLOGICAL_LIMITS:
        low, high = BIOLOGICAL_LIMITS[param]
        if val < low or val > high:
          return (
              True,
              "SENSOR_CONFLICT",
              f"Impossible {param} reading: {val} (Expected physical range: {low}-{high})",
          )

  # 3. Acute Hypoxia Emergency Check (DO < 3.0 mg/L)
  do_records = telemetry.get("DO", [])
  if not isinstance(do_records, list):
    do_records = [do_records]

  valid_do = [
      r["value"]
      for r in do_records
      if isinstance(r, dict) and r.get("value") is not None
  ]
  if valid_do and min(valid_do) < 3.0:
    species = session_state.get("species") or "pond stock"
    lowest = min(valid_do)
    return (
        True,
        "EMERGENCY_HYPOXIA",
        f"Critical hypoxia ({lowest} mg/L) detected for {species}.",
    )

  return False, None, None


def load_engine():
  print("[*] Loading Diagnostic Routing Models...")
  embedding_model = (
      embedding_functions.SentenceTransformerEmbeddingFunction(
          model_name="BAAI/bge-m3", device=DEVICE
      )
  )
  print(f"[*] Loading fine-tuned DeBERTa CrossEncoder from {NLI_MODEL_PATH}...")
  nli_model = CrossEncoder(NLI_MODEL_PATH, device=DEVICE)

  print(f"[*] Connecting to ChromaDB at {DB_PATH}...")
  client = chromadb.PersistentClient(path=DB_PATH)
  collection = client.get_collection(COLLECTION_NAME)

  print("[+] Diagnostic engine loaded successfully.")
  return embedding_model, nli_model, collection


def retrieve_evidence(
    query: str,
    embedding_model,
    collection,
    target_species: Optional[str] = None,
    target_parameter: Optional[str] = None,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieves candidate evidence chunks using:
    1. Species normalization (fish vs. shrimp vs. general).
    2. Deterministic topic isolation at the vector database level to prevent 
       pathology/chemical-treatment chunks from contaminating limnological queries.
    """
    embeddings = embedding_model([query])
    query_lower = query.lower()

    # 1. Normalize Species Facet
    species_facet = "general"
    if target_species:
        sp_lower = target_species.lower()
        if any(k in sp_lower for k in ["shrimp", "prawn", "vannamei", "monodon"]):
            species_facet = "shrimp"
        elif any(k in sp_lower for k in ["fish", "carp", "tilapia", "rohu", "catla", "mrigal"]):
            species_facet = "fish"
        elif sp_lower in ("fish", "shrimp"):
            species_facet = sp_lower

    # 2. Build Structural ChromaDB Filter
    # If the diagnostic state is evaluating water quality / dissolved oxygen,
    # restrict topic facets to water quality and pond management.
    conditions = []
    
    # Species condition
    if species_facet in ("fish", "shrimp"):
        conditions.append({"species": {"$in": [species_facet, "general"]}})
        
    # Topic condition: Isolate limnology from disease/parasitology
    if target_parameter in ("DO", "oxygen", "pH", "TAN", "ammonia", "salinity") or any(
        w in query_lower for w in ["oxygen", "aeration", "hypoxia", "do level"]
    ):
        conditions.append({"primary_topic": {"$in": ["water_quality", "general_management", "pond_preparation"]}})

    if len(conditions) == 1:
        where_clause = conditions[0]
    elif len(conditions) > 1:
        where_clause = {"$and": conditions}
    else:
        where_clause = None

    try:
        results = collection.query(
            query_embeddings=embeddings,
            n_results=top_k * 3,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        # Fallback to species-only if strict compound filtering fails
        fallback_where = {"species": {"$in": [species_facet, "general"]}} if species_facet in ("fish", "shrimp") else None
        results = collection.query(
            query_embeddings=embeddings,
            n_results=top_k * 3,
            where=fallback_where,
            include=["documents", "metadatas", "distances"],
        )

    evidence = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        text = results["documents"][0][i]
        metadata = results["metadatas"][0][i] or {}

        if distance > MAX_DISTANCE_THRESHOLD or len(text.strip()) < 25:
            continue

        evidence.append({
            "id": results["ids"][0][i],
            "text": text,
            "metadata": metadata,
            "distance": distance,
        })

        if len(evidence) >= top_k:
            break

    return evidence


def convert_query_to_declarative_hypothesis(query: str, session_state: Optional[Dict[str, Any]] = None) -> str:
  """Converts queries into declarative hypotheses for DeBERTa NLI cross-encoding."""
  q = query.strip()
  q_clean = q.rstrip("?. ")
  q_low = q_clean.lower()
  session_state = session_state or {}

  species = session_state.get("species") or "fish"
  telemetry = session_state.get("telemetry", {})
  symptoms = list(session_state.get("symptoms", []))

  # Elliptical or action-seeking questions ("what should i do", "what now", "how to treat")
  action_triggers = ["what should i do", "how to treat", "what now", "do about that", "remedy", "corrective action"]
  if any(trig in q_low for trig in action_triggers) or len(q_clean.split()) <= 6:
    if "DO" in telemetry or any("surface" in s or "hypoxia" in s for s in symptoms):
      return f"Aeration, water exchange, and feeding reduction remediate low dissolved oxygen and hypoxia distress in {species} ponds."
    if "TAN" in telemetry or "ammonia" in q_low:
      return f"Water exchange and biofloc management reduce toxic ammonia levels in {species} ponds."
    if "pH" in telemetry:
      return f"Lime and water treatments stabilize fluctuating pH levels in {species} aquaculture ponds."
    return f"Corrective aquaculture pond management and water quality restoration protect {species} health."

  # Informational queries
  wh_words = ("what", "how", "why", "when", "which", "is", "can", "should")
  first_word = q_low.split()[0] if q_low else ""

  if first_word in wh_words or q.endswith("?"):
    q_sub = re.sub(
        r"^(what are|what is|what should|what|how to|how do|how|why do|why|which are|which|can i|should i)\s+",
        "",
        q_clean,
        flags=re.IGNORECASE,
    )
    return f"This documentation explains {q_sub} in {species} ponds."

  return q


def evaluate_evidence(
    hypothesis: str,
    evidence_chunks: List[Dict[str, Any]],
    nli_model: CrossEncoder,
    session_state: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
  """Evaluates NLI probabilities using declarative proposition mapping."""
  if not evidence_chunks:
    return []

  declarative_claim = convert_query_to_declarative_hypothesis(hypothesis, session_state=session_state)
  pairs = [[chunk["text"], declarative_claim] for chunk in evidence_chunks]

  logits = nli_model.predict(pairs)
  evaluated = []

  for chunk, logit in zip(evidence_chunks, logits):
    probabilities = torch.softmax(torch.tensor(logit), dim=0).tolist()
    evaluated.append({
        **chunk,
        "contradiction": float(probabilities[0]),
        "entailment": float(probabilities[1]),
        "neutral": float(probabilities[2]),
        "declarative_hypothesis": declarative_claim,
    })

  return evaluated


def make_decision(evaluated_chunks: List[Dict[str, Any]]) -> str:
  if not evaluated_chunks:
    return "ABSTAIN"

  # Strong entailment check
  supporting = [
      c
      for c in evaluated_chunks
      if c.get("entailment", 0.0) >= ENTAILMENT_THRESHOLD
      and c.get("contradiction", 0.0) < CONTRADICTION_THRESHOLD
  ]
  if supporting:
    return "ANSWER"

  # Strong contradiction check
  contradicting = [
      c for c in evaluated_chunks if c.get("contradiction", 0.0) >= 0.70
  ]
  if contradicting:
    return "ABSTAIN"

  # Close semantic match fallback (BGE-M3 distance < 0.45 with moderate neutral confidence)
  top_chunk = evaluated_chunks[0]
  if (
      top_chunk.get("distance", 1.0) < 0.45
      and top_chunk.get("neutral", 0.0) >= NEUTRAL_THRESHOLD
  ):
    return "ANSWER"

  return "CLARIFY"


def run_diagnostic(
    search_prompt: str,
    session_state: Dict[str, Any],
    embedding_model,
    nli_model,
    collection,
    target_parameter: Optional[str] = None
) -> Dict[str, Any]:
    print(f"\n[?] Executing Diagnostic on: '{search_prompt}'")
    species = session_state.get("species")

    # Retrieval with Species and Topic Isolation
    evidence = retrieve_evidence(
        search_prompt, 
        embedding_model, 
        collection, 
        target_species=species,
        target_parameter=target_parameter
    )
    if not evidence:
        return {
            "hypothesis": search_prompt,
            "decision": "ABSTAIN",
            "evidence": [],
            "detail": "No relevant literature found.",
        }

    evaluated_chunks = evaluate_evidence(search_prompt, evidence, nli_model, session_state=session_state)
    decision = make_decision(evaluated_chunks)
    print(f"[+] Final Pipeline Decision: {decision}")

    return {
        "hypothesis": search_prompt,
        "decision": decision,
        "evidence": evaluated_chunks,
        "detail": None,
    }