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
# INVARIANT RC-01: 'probe' and 'meter' are hardware descriptors, NOT Dissolved Oxygen.
PARAMETER_LEMMA_MAP = {
    # Dissolved Oxygen
    "oxygen": "DO",
    "do": "DO",
    "d.o.": "DO",
    "d.o": "DO",
    "do.": "DO",
    "d0": "DO",
    "o2": "DO",
    "dissolved oxygen": "DO",
    "hypoxia": "DO",
    "ఆక్సిజన్": "DO",
    # pH / Acidity
    "ph": "pH",
    "acidity": "pH",
    "alkalinity": "alkalinity",
    "పిహెచ్": "pH",
    "ఆల్కలినిటీ": "alkalinity",
    "క్షారత": "alkalinity",
    # Nitrogenous Waste
    "ammonia": "TAN",
    "tan": "TAN",
    "nh3": "TAN",
    "nh4": "TAN",
    "nitrite": "NO2",
    "nitrate": "NO3",
    "అమ్మోనియా": "TAN",
    # Physical / Environmental
    "temperature": "temperature",
    "temp": "temperature",
    "c": "temperature",
    "°c": "temperature",
    "celsius": "temperature",
    "salinity": "salinity",
    "ppt": "salinity",
    "transparency": "transparency",
    "turbidity": "turbidity",
    "depth": "depth",
    "orp": "ORP",
    "లవణీయత": "salinity",
    "ఉష్ణోగ్రత": "temperature"
}

# Ambiguous hardware identifiers that should never be mapped directly to a parameter
AMBIGUOUS_HARDWARE_TOKENS = {"probe", "meter", "sensor", "kit"}

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

TEMPORAL_MODIFIERS = {
    "morning", "dawn", "afternoon", "evening", "night", "dusk", "noon",
    "am", "pm", "gantalaki", "గంటలకు"
}

# Tokens that indicate a number is a timestamp or hour, not water telemetry
TIME_INDICATORS = {"am", "pm", "gantalaki", "గంటలకు", "hours", "hrs", "o'clock", "clock"}


OPERATIONAL_METADATA_TOKENS = {
    "acre", "acres", "ha", "hectare", "hectares", "feet", "ft", "inch", "inches",
    "doc", "day", "days", "d", "tray", "trays", "bag", "bags", "kg", "kgs",
    "hr", "hrs", "hour", "hours", "min", "mins", "minute", "minutes",
    "pond", "tank", "count", "nos", "pcs", "batch", "tanki"
}

def is_timestamp_token(token: spacy.tokens.Token) -> bool:
    """
    INVARIANT RC-01 & STRESS-P0:
    Detects if a numeric token represents time, clock markers, or operational farm metadata
    (e.g., '5:30', '4:45am', 'DOC 78', '4 trays', '3 acre', '1.5 hr').
    """
    raw_text = token.text.lower().strip(".,()[]{}")
    
    # 1. Regex check for unspaced colon timestamps (e.g., 5:30, 04:45am, 17:00)
    if re.match(r"^\d{1,2}:\d{2}(?:am|pm)?$", raw_text):
        return True

    doc = token.doc
    
    # 2. Check forward token for timestamp or operational unit indicators
    if token.i + 1 < len(doc):
        next_t = doc[token.i + 1].text.lower().rstrip(".,:;()[]")
        if next_t in TIME_INDICATORS or next_t in OPERATIONAL_METADATA_TOKENS:
            return True
            
    # 3. Check for spaced colon time constructs (e.g., [5] [:] [30])
    if token.i + 2 < len(doc) and doc[token.i + 1].text in (":", "."):
        if doc[token.i + 2].like_num:
            return True

    # 4. Check backward token for operational unit prefixes (e.g., 'DOC 78', 'pond 4', 'tank 2', 'count 40')
    if token.i > 0:
        prev_t = doc[token.i - 1].text.lower().lstrip(".,:;()[]")
        if prev_t in {"doc", "pond", "tank", "tray", "trays", "count", "no", "nos"}:
            return True

    return False


def extract_temporal_aspect(token: spacy.tokens.Token) -> Optional[str]:
    """Finds temporal words (morning, afternoon, etc.) near the measurement token."""
    start = max(0, token.i - 6)
    end = min(len(token.doc), token.i + 6)
    for t in token.doc[start:end]:
        t_clean = t.text.lower().rstrip(".,:")
        if t_clean in TEMPORAL_MODIFIERS:
            return t_clean
    return None


def extract_telemetry(doc: spacy.tokens.Doc, pending_metrics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    telemetry = []
    claimed_tokens = set()
    last_seen_param = None

    for token in doc:
        token_lemma = token.lemma_.lower().rstrip(":")
        token_text = token.text.lower().rstrip(":")
        
        # Track active parameter keyword in sentence, ignoring ambiguous hardware words
        if token_lemma in PARAMETER_LEMMA_MAP:
            last_seen_param = PARAMETER_LEMMA_MAP[token_lemma]
        elif token_text in PARAMETER_LEMMA_MAP:
            last_seen_param = PARAMETER_LEMMA_MAP[token_text]

        if token.like_num or token.pos_ == "NUM":
            # INVARIANT RC-01: Ignore numbers that represent time/hours (e.g., '4 am', '6 am')
            if is_timestamp_token(token):
                continue

            cleaned = re.sub(r"[^\d.]", "", token.text)
            if not cleaned:
                continue
            try:
                val = float(cleaned)
            except ValueError:
                continue

            # 1. Syntactic dependency lookup
            param_key = find_governing_parameter(token)

            # 2. Forward window unit deduction
            next_token = doc[token.i + 1] if token.i + 1 < len(doc) else None
            next_raw = next_token.text.lower().rstrip(".,:;") if next_token else ""
            
            window_slice = doc[token.i + 1 : min(len(doc), token.i + 5)]
            forward_text = "".join(t.text.lower().rstrip(".,:;") for t in window_slice)

            detected_unit = ""
            inferred_param_from_unit = None

            # Check unambiguous units first (bijective mappings)
            if forward_text.startswith(("ppt", "psu")) or next_raw in ("ppt", "psu"):
                detected_unit = "ppt"
                inferred_param_from_unit = "salinity"
            elif next_raw in ("c", "°c", "celsius", "deg") or forward_text.startswith(("°c", "celsius")):
                detected_unit = "°C"
                inferred_param_from_unit = "temperature"
            elif forward_text.startswith(("mg/l", "ppm", "mg/lt")) or next_raw in ("mg/l", "ppm"):
                detected_unit = "mg/L"
                if not param_key and not last_seen_param:
                    inferred_param_from_unit = "DO"

            # --- INVARIANT P0: BIJECTIVE PHYSICAL UNIT INVARIANCE ---
            # Bijective physical units (ppt -> salinity, °C -> temperature) mathematically
            # cannot represent other parameters, overriding any long-range parse misattachments.
            if inferred_param_from_unit in ("salinity", "temperature"):
                param_key = inferred_param_from_unit
                unit = detected_unit
            elif param_key in ("pH", "temperature", "salinity") and detected_unit in ("mg/L", "ppm"):
                # Concentration units contradict pH or temperature; re-route to concentration
                param_key = last_seen_param if last_seen_param in ("DO", "TAN", "NO2", "NO3", "alkalinity") else "DO"
                unit = detected_unit
            elif param_key:
                unit = detected_unit or DEFAULT_UNITS.get(param_key, "")
            elif inferred_param_from_unit:
                param_key = inferred_param_from_unit
                unit = detected_unit
            else:
                unit = detected_unit

            # 3. Proximity fallback: Scan ±5 tokens if still unknown
            if not param_key:
                for offset in (-1, -2, -3, -4, -5, 1, 2):
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

            # 4. Context inheritance fallback (only if NO contradictory unit was detected)
            if not param_key and last_seen_param:
                if last_seen_param == "pH" and 3.5 <= val <= 11.5:
                    param_key = "pH"
                elif last_seen_param in ("DO", "TAN", "NO2", "NO3", "alkalinity"):
                    # Only inherit concentration parameters if unit is compatible
                    if not unit or unit == "mg/L":
                        param_key = last_seen_param
                elif last_seen_param == "temperature" and 0.0 <= val <= 50.0:
                    if not unit or unit == "°C":
                        param_key = "temperature"
                elif last_seen_param == "salinity" and 0.0 <= val <= 70.0:
                    if not unit or unit == "ppt":
                        param_key = "salinity"

            if param_key:
                if not unit:
                    unit = DEFAULT_UNITS.get(param_key, "")
                temporal = extract_temporal_aspect(token) or "current"
                telemetry.append({
                    "parameter": param_key,
                    "value": val,
                    "unit": unit,
                    "temporal": temporal
                })
                claimed_tokens.add(token.i)

    # Pass 2: Conversational Slot-Filling for Naked Numbers
    if pending_metrics:
        unclaimed_numbers = []
        for token in doc:
            if token.i not in claimed_tokens and (token.like_num or token.pos_ == "NUM"):
                if is_timestamp_token(token):
                    continue
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
                "unit": DEFAULT_UNITS.get(canonical, ""),
                "temporal": None
            })

    return telemetry

def extract_temporal_aspect(token: spacy.tokens.Token) -> Optional[str]:
    """Finds temporal words (morning, afternoon, etc.) near the measurement token."""
    start = max(0, token.i - 6)
    end = min(len(token.doc), token.i + 6)
    for t in token.doc[start:end]:
        t_clean = t.text.lower()
        if t_clean in TEMPORAL_MODIFIERS:
            return t_clean
    return None

def find_governing_parameter(num_token: spacy.tokens.Token) -> Optional[str]:
    """
    Syntactically navigates the dependency tree to find the governing parameter.
    Handles:
    1. Direct head attachment (e.g., 'pH 7.1')
    2. Copular predicative structures (e.g., 'pH is 7.1')
    3. Verb prepositional complements (e.g., 'pH rises to 9.4', 'DO drops to 1.8')
    4. Unit-attribute structures (e.g., 'temperature is 31 C' -> num points to C, C points to is)
    """
    head = num_token.head
    head_lemma = head.lemma_.lower().rstrip(":")
    if head_lemma in PARAMETER_LEMMA_MAP:
        return PARAMETER_LEMMA_MAP[head_lemma]

    # Copular / Auxiliary constructions ('pH is 7.1')
    if head.pos_ in ("AUX", "VERB"):
        for child in head.children:
            child_lemma = child.lemma_.lower().rstrip(":")
            if child_lemma in PARAMETER_LEMMA_MAP:
                return PARAMETER_LEMMA_MAP[child_lemma]
            # Check compound subject (e.g., 'Water temperature is 31')
            for subchild in child.children:
                sub_lemma = subchild.lemma_.lower().rstrip(":")
                if sub_lemma in PARAMETER_LEMMA_MAP:
                    return PARAMETER_LEMMA_MAP[sub_lemma]

    # Prepositional phrases attached to verbs ('rises to 9.4')
    if head.dep_ == "prep" and head.head.pos_ in ("VERB", "AUX"):
        verb = head.head
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubjpass", "dobj"):
                child_lemma = child.lemma_.lower().rstrip(":")
                if child_lemma in PARAMETER_LEMMA_MAP:
                    return PARAMETER_LEMMA_MAP[child_lemma]

    # Unit symbol / noun attribute link (e.g. '31 C', '28 ppt' where 31 points to C, C points to 'is')
    if head.dep_ in ("attr", "pobj", "dobj") and head.head.pos_ in ("AUX", "VERB"):
        parent_verb = head.head
        for child in parent_verb.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                child_lemma = child.lemma_.lower().rstrip(":")
                if child_lemma in PARAMETER_LEMMA_MAP:
                    return PARAMETER_LEMMA_MAP[child_lemma]
                for subchild in child.children:
                    sub_lemma = subchild.lemma_.lower().rstrip(":")
                    if sub_lemma in PARAMETER_LEMMA_MAP:
                        return PARAMETER_LEMMA_MAP[sub_lemma]

    # Ancestor traversal (up to 3 levels)
    curr = head
    depth = 0
    while curr != curr.head and depth < 3:
        curr_lemma = curr.lemma_.lower().rstrip(":")
        if curr_lemma in PARAMETER_LEMMA_MAP:
            return PARAMETER_LEMMA_MAP[curr_lemma]
        for sibling in curr.children:
            sib_lemma = sibling.lemma_.lower().rstrip(":")
            if sib_lemma in PARAMETER_LEMMA_MAP:
                return PARAMETER_LEMMA_MAP[sib_lemma]
        curr = curr.head
        depth += 1

    return None

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

    # Store query under all common context keys
    session_state["query"] = farmer_query
    session_state["raw_query"] = farmer_query
    session_state["last_query"] = cleaned_query

    doc = nlp(cleaned_query)

    # 1. Species & Habitat Isolation
    organism, habitat = extract_organism_and_habitat(doc, prior_organism=session_state.get("species"))
    if organism:
        session_state["species"] = organism
    if habitat:
        session_state["habitat"] = habitat

    # 2. Telemetry Extraction
    pending = session_state.get("pending_metrics")
    extracted_telemetry = extract_telemetry(doc, pending_metrics=pending)
    
    # Track whether new telemetry was provided in this specific turn
    session_state["fresh_telemetry_in_turn"] = bool(extracted_telemetry)

    if extracted_telemetry:
        session_state["pending_metrics"] = None
        if "telemetry" not in session_state or not isinstance(session_state["telemetry"], dict):
            session_state["telemetry"] = {}

        # Group metrics extracted strictly within THIS turn, PRESERVING temporal aspect
        current_turn_metrics = {}
        for item in extracted_telemetry:
            p_name = item["parameter"]
            record = {
                "value": item["value"],
                "unit": item["unit"],
                "temporal": item.get("temporal")  # <-- Preserved for diurnal/conflict invariants
            }
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

    # Check for diurnal pH dynamics (T13)
    ph_records = session_state.get("telemetry", {}).get("pH", [])
    valid_ph = [r["value"] for r in ph_records if isinstance(r, dict) and r.get("value") is not None]
    is_diurnal_ph = len(valid_ph) >= 2 and (max(valid_ph) - min(valid_ph)) >= 1.5

    if is_diurnal_ph:
        status_phrase = "diurnal pH fluctuation and afternoon phytoplankton carbon dioxide depletion"
    elif has_explicit_symptoms:
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
                temp_label = f" ({r['temporal']})" if r.get("temporal") else ""
                telemetry_parts.append(f"{param_name}{temp_label} {r['value']} {r['unit']}".strip())
        elif isinstance(records, dict):
            telemetry_parts.append(f"{param_name} {records['value']} {records['unit']}".strip())
        
    telemetry_str = f" with {', '.join(telemetry_parts)}" if telemetry_parts else ""
    search_prompt = f"{target_entity.capitalize()} {status_phrase}{telemetry_str}."
    
    return search_prompt, session_state