"""
AquaAssist: Linguistic Semantic Frame & Clinical Intent Extractor (Hardened)
Deterministic extraction of telemetry, clinical signs, species, and habitat.
Stores structured list-based telemetry in session_state for conflict and emergency evaluation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Recognized limnological and water-quality concept roots
PARAMETER_LEMMA_MAP = {
    # Dissolved Oxygen
    "oxygen": "DO",
    "do": "DO",
    "d.o.": "DO",
    "d0": "DO",
    "o2": "DO",
    "dissolved oxygen": "DO",
    "probe": "DO",  
    "meter": "DO",  
    "hypoxia": "DO",
    # pH / Acidity
    "ph": "pH",
    "acidity": "pH",
    "alkalinity": "alkalinity",
    # Nitrogenous Waste
    "ammonia": "TAN",
    "tan": "TAN",
    "nh3": "TAN",
    "nh4": "TAN",
    "nitrite": "NO2",
    "nitrate": "NO3",
    # Physical / Environmental
    "temperature": "temperature",
    "temp": "temperature",
    "salinity": "salinity",
    "transparency": "transparency",
    "turbidity": "turbidity",
    "depth": "depth",
    "orp": "ORP"
}

UNIT_NORMALIZATION = {
    "ppm": "mg/L",
    "mg/l": "mg/L",
    "mg/lt": "mg/L",
    "ppt": "ppt",
    "psu": "ppt",
    "c": "°C",
    "celsius": "°C",
    "deg": "°C",
    "cm": "cm",
    "meter": "m",
    "m": "m",
    "mv": "mV"
}

DEFAULT_UNITS = {
    "DO": "mg/L",
    "pH": "",
    "TAN": "mg/L",
    "NO2": "mg/L",
    "NO3": "mg/L",
    "alkalinity": "mg/L",
    "temperature": "°C",
    "salinity": "ppt",
    "transparency": "cm",
    "depth": "m",
    "ORP": "mV"
}


def normalize_input(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r'[\r\n\t]+', ' ', text).strip()


def extract_organism_and_habitat(doc: spacy.tokens.Doc, prior_organism: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    text_lower = doc.text.lower()
    organism = None
    habitat = None

    fish_keywords = {"fish", "carp", "tilapia", "rohu", "catla", "catfish", "murrel", "చేపలు", "చేప"}
    shrimp_keywords = {"shrimp", "prawn", "vannamei", "monodon", "రొయ్యలు", "రొయ్య"}

    has_fish = any(re.search(rf'\b{re.escape(w)}\b', text_lower) for w in fish_keywords)
    has_shrimp = any(re.search(rf'\b{re.escape(w)}\b', text_lower) for w in shrimp_keywords)

    if has_fish and not has_shrimp:
        organism = "fish"
    elif has_shrimp and not has_fish:
        organism = "shrimp"
    elif prior_organism in ("fish", "shrimp"):
        organism = prior_organism

    if any(k in text_lower for k in ["freshwater", "fresh water", "మంచినీటి", "తీపి నీరు"]):
        habitat = "freshwater"
    elif any(k in text_lower for k in ["brackish", "saline", "salinity", "ఉప్పు నీరు"]):
        habitat = "brackish"

    return organism, habitat


def check_token_or_compounds(token: spacy.tokens.Token) -> Optional[str]:
    """Helper to check if a token or any of its compound modifiers represent a parameter."""
    t_lemma = token.lemma_.lower()
    if t_lemma in PARAMETER_LEMMA_MAP:
        return PARAMETER_LEMMA_MAP[t_lemma]
    for child in token.children:
        if child.dep_ == "compound" and child.lemma_.lower() in PARAMETER_LEMMA_MAP:
            return PARAMETER_LEMMA_MAP[child.lemma_.lower()]
    return None


def find_governing_parameter(token: spacy.tokens.Token) -> Optional[str]:
    """
    Traverses syntactic dependency links upward from a numeric token to identify
    its governing parameter noun, resolving prepositional and adverbial chains.
    """
    # 1. Direct noun head: "2.2 mg/L DO", "level of 2.2"
    head = token.head
    direct = check_token_or_compounds(head)
    if direct:
        return direct

    # 2. Traverse prepositional / adverbial chains ("dropped down to 2.2", "is at 2.2")
    curr = head
    while curr.pos_ in ("ADP", "ADV", "PART") and curr.head != curr:
        curr = curr.head

    # Check if the climbed head is directly a parameter or a predicate governing a parameter subject
    climbed = check_token_or_compounds(curr)
    if climbed:
        return climbed

    if curr.pos_ in ("VERB", "AUX"):
        for child in curr.children:
            if child.dep_ in ("nsubj", "nsubjpass", "dobj", "attr"):
                param = check_token_or_compounds(child)
                if param:
                    return param

    return None


def extract_telemetry(doc: spacy.tokens.Doc, pending_metrics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    telemetry = []
    claimed_tokens = set()

    for token in doc:
        if token.like_num or token.pos_ == "NUM":
            cleaned = re.sub(r"[^\d.]", "", token.text)
            if not cleaned:
                continue
            try:
                val = float(cleaned)
            except ValueError:
                continue

            param_key = find_governing_parameter(token)

            # Proximity fallback: Scan ±4 tokens for parameter keywords
            if not param_key:
                for offset in (-1, -2, -3, -4, 1, 2):
                    idx = token.i + offset
                    if 0 <= idx < len(doc):
                        w = doc[idx].lemma_.lower().rstrip(":")
                        if w in PARAMETER_LEMMA_MAP:
                            param_key = PARAMETER_LEMMA_MAP[w]
                            break
                        raw_w = doc[idx].text.lower().rstrip(":")
                        if raw_w in PARAMETER_LEMMA_MAP:
                            param_key = PARAMETER_LEMMA_MAP[raw_w]
                            break

            # Look ahead across tokens to reconstruct split units (e.g., ['mg', '/', 'l'] -> 'mg/l')
            window_slice = doc[token.i + 1 : min(len(doc), token.i + 5)]
            forward_text = "".join(t.text.lower() for t in window_slice)

            # Unit-deduction fallback (resolves "reads 7.2 mg/L" when DO is omitted)
            if not param_key:
                if forward_text.startswith(("mg/l", "ppm", "mg/lt", "mg")):
                    param_key = "DO"
                elif forward_text.startswith(("ppt", "psu")):
                    param_key = "salinity"
                elif forward_text.startswith(("c", "°c", "celsius")):
                    param_key = "temperature"

            if param_key:
                unit = DEFAULT_UNITS.get(param_key, "")
                if forward_text.startswith(("mg/l", "ppm", "mg/lt")):
                    unit = "mg/L"
                elif forward_text.startswith(("ppt", "psu")):
                    unit = "ppt"
                elif forward_text.startswith(("c", "°c", "celsius")):
                    unit = "°C"

                telemetry.append({
                    "parameter": param_key,
                    "value": val,
                    "unit": unit
                })
                claimed_tokens.add(token.i)

    # Pass 2: Conversational Slot-Filling for Naked Numbers
    if pending_metrics:
        unclaimed_numbers = []
        for token in doc:
            if token.i not in claimed_tokens and (token.like_num or token.pos_ == "NUM"):
                cleaned = re.sub(r"[^\d.]", "", token.text)
                if cleaned:
                    try:
                        unclaimed_numbers.append(float(cleaned))
                    except ValueError:
                        pass

        for metric, val in zip(pending_metrics, unclaimed_numbers):
            canonical = PARAMETER_LEMMA_MAP.get(metric.lower(), metric)
            telemetry.append({
                "parameter": canonical,
                "value": val,
                "unit": DEFAULT_UNITS.get(canonical, "")
            })

    return telemetry


def extract_clinical_symptoms(doc: spacy.tokens.Doc) -> List[str]:
    text_lower = doc.text.lower()
    symptoms = []

    # 1. Anorexia / Off-feed
    for token in doc:
        if token.lemma_ in ("eat", "feed", "consume"):
            has_neg = any(child.dep_ == "neg" or child.text.lower() in ("no", "not", "stop", "stopped", "lack") 
                          for child in token.children)
            if token.head.lemma_ in ("stop", "halt", "cease"):
                has_neg = True

            if has_neg or "తినడం లేదు" in text_lower or "ఆహారం తీసుకోవడం లేదు" in text_lower:
                symptoms.append("anorexia / off-feed")

    # 2. Respiratory distress / Surface gasping
    gasp_terms = ["gasp", "gasping", "piping", "gulping", "surface gasping", "swimming at surface", "at the surface", "పైకి తేలడం"]
    if any(term in text_lower for term in gasp_terms):
        symptoms.append("surface gasping / hypoxia distress")

    # 3. Abnormal locomotion
    lethargy_terms = ["erratic", "sluggish", "lethargic", "spinning", "unable to swim", "ఈదలేకపోవడం"]
    if any(term in text_lower for term in lethargy_terms):
        symptoms.append("sluggish / erratic swimming")

    # 4. Acute Mortality
    mortality_terms = ["dying", "dead", "mortality", "losses", "floating dead", "చనిపోతున్నాయి", "మరణాలు"]
    if any(re.search(rf'\b{re.escape(term)}\b', text_lower) for term in mortality_terms):
        symptoms.append("acute mortality")

    # 5. Visual water symptoms
    if re.search(r'\b(dark green|pea green|turbid|foul smell|bad odor)\b', text_lower):
        symptoms.append("abnormal water discoloration/odor")

    return symptoms


def is_informational_query(doc: spacy.tokens.Doc) -> bool:
    text_lower = doc.text.lower()
    question_triggers = [
        "what", "how", "why", "when", "which", "causes", "reason", "procedure", 
        "recommend", "optimal", "ideal", "parameters", "range", "monitor",
        "ఏమిటి", "ఎలా", "ఎందుకు", "ఎంత"
    ]
    first_tokens = [tok.text.lower() for tok in doc[:3]]
    if any(trig in first_tokens for trig in question_triggers):
        return True
    return any(re.search(rf'\b{re.escape(trig)}\b', text_lower) for trig in question_triggers)


def extract_hypothesis(farmer_query: str, session_state: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    cleaned_query = normalize_input(farmer_query)
    
    if session_state is None:
        session_state = {}
    if "telemetry" not in session_state:
        session_state["telemetry"] = {}
    if "symptoms" not in session_state:
        session_state["symptoms"] = []

    doc = nlp(cleaned_query)

    # 1. Species & Habitat Isolation
    organism, habitat = extract_organism_and_habitat(doc, prior_organism=session_state.get("species"))
    if organism:
        session_state["species"] = organism
    if habitat:
        session_state["habitat"] = habitat

    # 2. Telemetry Extraction (Appends to lists to support T11 conflict detection)
    pending = session_state.get("pending_metrics")
    extracted_telemetry = extract_telemetry(doc, pending_metrics=pending)
    
    # Track whether new telemetry was provided in this specific turn
    session_state["fresh_telemetry_in_turn"] = bool(extracted_telemetry)

    if extracted_telemetry:
        session_state["pending_metrics"] = None
        if "telemetry" not in session_state or not isinstance(session_state["telemetry"], dict):
            session_state["telemetry"] = {}

        # Group metrics extracted strictly within THIS turn
        current_turn_metrics = {}
        for item in extracted_telemetry:
            p_name = item["parameter"]
            record = {"value": item["value"], "unit": item["unit"]}
            if p_name not in current_turn_metrics:
                current_turn_metrics[p_name] = []
            current_turn_metrics[p_name].append(record)

        # Overwrite previous turn readings with fresh turn readings (fixes T11 persistent state loop)
        for p_name, records in current_turn_metrics.items():
            session_state["telemetry"][p_name] = records
    
    # 3. Clinical Symptoms
    symptoms = extract_clinical_symptoms(doc)
    if symptoms:
        session_state["symptoms"] = set(symptoms)

    is_info = is_informational_query(doc)
    session_state["last_query"] = cleaned_query

    # --- RETRIEVAL QUERY FORMULATION ---
    # Informational Questions (T1, T2, T3, T5)
    if is_info or (not symptoms and not extracted_telemetry):
        context_tags = []
        if session_state.get("species"):
            context_tags.append(session_state["species"])
        if session_state.get("habitat"):
            context_tags.append(session_state["habitat"])
            
        qualifier = f" ({', '.join(context_tags)})" if context_tags else ""
        search_prompt = f"{cleaned_query}{qualifier}"
        return search_prompt, session_state

    # Diagnostic / Telemetry Prompt Formulation
    target_entity = session_state.get("species") or "fish and shrimp"
    has_explicit_symptoms = bool(session_state.get("symptoms"))
    
    # Check if DO is optimal (>= 5.0 mg/L)
    do_records = session_state.get("telemetry", {}).get("DO", [])
    valid_do = [r["value"] for r in do_records if isinstance(r, dict) and r.get("value") is not None]
    is_healthy_do = bool(valid_do and min(valid_do) >= 5.0)

    if has_explicit_symptoms:
        symptom_str = ", ".join(session_state["symptoms"])
        status_phrase = f"showing {symptom_str}"
    elif is_healthy_do:
        status_phrase = "optimal dissolved oxygen water quality management"
    else:
        status_phrase = "water quality management"
    
    telemetry_parts = []
    for param_name, records in session_state.get("telemetry", {}).items():
        if isinstance(records, list):
            for r in records:
                telemetry_parts.append(f"{param_name} {r['value']} {r['unit']}".strip())
        elif isinstance(records, dict):
            telemetry_parts.append(f"{param_name} {records['value']} {records['unit']}".strip())
        
    telemetry_str = f" with {', '.join(telemetry_parts)}" if telemetry_parts else ""
    search_prompt = f"{target_entity.capitalize()} {status_phrase}{telemetry_str}."
    
    return search_prompt, session_state