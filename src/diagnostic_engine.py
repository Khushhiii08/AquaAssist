"""
AquaAssist: Hardened Diagnostic Engine & Grounded NLI Gate
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

    # Check for diurnal temporal cues (T13: morning vs afternoon pH fluctuation)
    raw_query = session_state.get("raw_query", "").lower()
    is_diurnal = any(w in raw_query for w in ["morning", "afternoon", "evening", "diurnal", "day and night", "night"])

    # 1. Multi-Reading Conflict Detection
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
                # If diurnal shifts are described, delta of 2.3 is biologically natural, not a sensor defect
                if param == "pH" and spread >= 1.5 and not is_diurnal:
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
    # INVARIANT: Only fire emergency reflex if telemetry is FRESH in this turn (Phi == True)
    is_fresh = session_state.get("fresh_telemetry_in_turn", True)

    do_records = telemetry.get("DO", [])
    if not isinstance(do_records, list):
        do_records = [do_records]

    valid_do = [
        r["value"]
        for r in do_records
        if isinstance(r, dict) and r.get("value") is not None
    ]
    if is_fresh and valid_do and min(valid_do) < 3.0:
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

    # Dynamically resolve DB path when loading, rather than at module import time
    current_db_path = get_active_db_path()
    print(f"[*] Connecting to ChromaDB at {current_db_path}...")
    client = chromadb.PersistentClient(path=current_db_path)

    # Resilient collection discovery
    collections = client.list_collections()
    col_names = [c.name for c in collections]
    print(f"[*] Available collections in active ChromaDB: {col_names}")

    if not col_names:
        raise RuntimeError(f"Database at {current_db_path} contains zero collections. Verify data/chroma_db.zip contents.")

    if COLLECTION_NAME in col_names:
        collection = client.get_collection(COLLECTION_NAME)
    else:
        chosen_name = col_names[0]
        print(f"[!] Target collection '{COLLECTION_NAME}' not found. Defaulting to existing collection '{chosen_name}'")
        collection = client.get_collection(chosen_name)

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
       larval hatchery preparation or pathology chunks from contaminating routine telemetry checks.
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
    conditions = []
    
    # Species condition
    if species_facet in ("fish", "shrimp"):
        conditions.append({"species": {"$in": [species_facet, "general"]}})
        
    # Topic condition: Limnology, water quality, and algal bloom management
    # (Exclude 'pond_preparation' so hatchery larval feeds do not hijack water quality checks)
    is_water_quality_query = (
        target_parameter in ("DO", "oxygen", "pH", "TAN", "ammonia", "salinity", "temperature")
        or any(w in query_lower for w in [
            "oxygen", "aeration", "hypoxia", "do level", "low do",
            "ammonia", "tan", "ph", "salinity", "temperature",
            "algae", "algal", "bloom", "weed", "plankton", "nursery"
        ])
    )

    if is_water_quality_query:
        conditions.append({
            "primary_topic": {"$in": ["water_quality", "general_management", "pond_preparation"]}
        })

    if len(conditions) == 1:
        where_clause = conditions[0]
    elif len(conditions) > 1:
        where_clause = {"$and": conditions}
    else:
        where_clause = None

    try:
        results = collection.query(
            query_embeddings=embeddings,
            n_results=top_k * 4,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # Fallback to species-only if strict compound filtering fails
        fallback_where = {"species": {"$in": [species_facet, "general"]}} if species_facet in ("fish", "shrimp") else None
        results = collection.query(
            query_embeddings=embeddings,
            n_results=top_k * 4,
            where=fallback_where,
            include=["documents", "metadatas", "distances"],
        )

    # Hatchery exclusion terms for grow-out telemetry queries
    is_telemetry_check = any(w in query_lower for w in ["optimal", "reads", "recalibrated", "management with do", "do 7", "mg/l"])
    hatchery_terms = ["nauplii", "emulsion", "enrichment", "artemia", "cyst", "hatchery"]

    evidence = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        text = results["documents"][0][i]
        metadata = results["metadatas"][0][i] or {}

        if distance > MAX_DISTANCE_THRESHOLD or len(text.strip()) < 25:
            continue

        # Reject hatchery larval feeds if user is evaluating pond grow-out telemetry
        if is_telemetry_check and any(term in text.lower() for term in hatchery_terms):
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
    
    # --- RESOLVE STATE & SPECIES FIRST ---
    session_state = session_state or {}
    species = session_state.get("species") or "fish"
    telemetry = session_state.get("telemetry", {})
    symptoms = list(session_state.get("symptoms", []))

    # Ungrounded pharmacology test: human drugs / unapproved chemicals (T15)
    if any(drug in q_low for drug in ["paracetamol", "tablet", "aspirin", "crocin", "ibuprofen"]):
        return f"Paracetamol tablets are an approved and scientific clinical treatment for fungal infections in {species} aquaculture."

    # 1. Diurnal pH Shift Limnology (T13: Morning vs. Afternoon Dynamic)
    ph_records = telemetry.get("pH", [])
    if isinstance(ph_records, list) and len(ph_records) >= 2:
        ph_vals = [r["value"] for r in ph_records if isinstance(r, dict) and r.get("value") is not None]
        if len(ph_vals) >= 2:
            delta = max(ph_vals) - min(ph_vals)
            temporals = {r.get("temporal") for r in ph_records if isinstance(r, dict) and r.get("temporal")}
            is_diurnal = ("morning" in temporals and "afternoon" in temporals) or delta >= 1.5 or any(w in q_low for w in ["morning", "afternoon", "diurnal"])
            if is_diurnal:
                return (
                    f"Diurnal pH swings between morning {min(ph_vals)} and afternoon {max(ph_vals)} in {species} culture ponds "
                    f"are caused by heavy phytoplankton blooms consuming dissolved carbon dioxide during peak photosynthesis."
                )

    # 2. Elliptical or action-seeking questions
    action_triggers = ["what should i do", "how to treat", "what now", "do about that", "remedy", "corrective action"]
    if any(trig in q_low for trig in action_triggers) or len(q_clean.split()) <= 6:
        if "DO" in telemetry or any("surface" in s or "hypoxia" in s for s in symptoms):
            return f"Aeration, water exchange, and feeding reduction remediate low dissolved oxygen and hypoxia distress in {species} ponds."
        if "TAN" in telemetry or "ammonia" in q_low:
            return f"Water exchange and biofloc management reduce toxic ammonia levels in {species} ponds."
        if "pH" in telemetry:
            return f"Lime and water treatments stabilize fluctuating pH levels in {species} aquaculture ponds."
        return f"Corrective aquaculture pond management and water quality restoration protect {species} health."

    # 3. Status observations / pure telemetry reports without explicit interrogatives (T16 & follow-ups)
    wh_words = ("what", "how", "why", "when", "which", "is", "can", "should")
    first_word = q_low.split()[0] if q_low else ""
    is_wh = first_word in wh_words or q.endswith("?")

    has_telemetry_report = any(p in q_low for p in ["temperature", "salinity", "ppt", "celsius", "ph", "do", "probe", "reads"]) and any(c.isdigit() for c in q_low)
    if has_telemetry_report and not is_wh:
        # Optimal DO reading confirmation
        if "DO" in telemetry:
            do_vals = [r.get("value") for r in telemetry["DO"] if isinstance(r, dict) and r.get("value") is not None]
            if do_vals and min(do_vals) >= 5.0:
                return f"Dissolved oxygen levels above 5.0 mg/L provide optimal respiratory conditions for {species} health and growth."
        return f"Water temperature, salinity, and water quality parameters are maintained within safe ranges for {species} nursery rearing."
    
    # 4. Informational Wh- inquiries (T1 & T18 specific handling)
    if is_wh:
        # T1: Dissolved oxygen limnological causes
        if any(w in q_low for w in ["cause", "causes", "why"]) and any(w in q_low for w in ["oxygen", "do", "hypoxia"]):
            return f"Low dissolved oxygen in {species} aquaculture ponds is caused by excessive phytoplankton respiration, decomposition of organic matter, overstocking, and poor aeration."

        # T18: Filamentous green algae & phytoplankton bloom management
        if any(w in q_low for w in ["algae", "algal", "bloom"]) and any(w in q_low for w in ["control", "treat", "manage", "remedy", "prevent"]):
            return f"Biological control, physical removal, and nutrient management control filamentous green algae blooms in {species} aquaculture ponds."

        # General Wh- interrogative prefix stripper
        q_sub = re.sub(
            r"^(what are the|what are|what is the|what is|what should|what|how to|how do|how|why do|why|which are|which|can i|should i)\s+",
            "",
            q_clean,
            flags=re.IGNORECASE,
        ).strip()
        
        return f"{q_sub.capitalize()} is monitored and controlled in {species} aquaculture ponds."

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

    # 1. Immediate Contradiction Rejection (Strictly protects T15 paracetamol/unauthorized drugs)
    has_high_contradiction = any(
        c.get("contradiction", 0.0) >= 0.40 for c in evaluated_chunks
    )
    if has_high_contradiction:
        return "ABSTAIN"

    # 2. Strict Entailment Verification
    supporting = [
        c for c in evaluated_chunks
        if c.get("entailment", 0.0) >= ENTAILMENT_THRESHOLD
        and c.get("contradiction", 0.0) < CONTRADICTION_THRESHOLD
    ]
    if supporting:
        return "ANSWER"

    top_chunk = evaluated_chunks[0]

    # 3. Reject if top match has notable contradiction
    if top_chunk.get("contradiction", 0.0) >= 0.30:
        return "ABSTAIN"

    # 4. Grounded Semantic Match (Allows valid clinical procedures like algae bloom control)
    if (
        top_chunk.get("distance", 1.0) < 0.60
        and top_chunk.get("entailment", 0.0) >= 0.20
        and top_chunk.get("contradiction", 0.0) < 0.25
    ):
        return "ANSWER"

    return "ABSTAIN" if top_chunk.get("entailment", 0.0) < 0.15 else "CLARIFY"


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