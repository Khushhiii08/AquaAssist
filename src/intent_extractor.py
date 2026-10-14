import re

# Phase 1: Entity Resolution Layer
CANONICAL_PARAMS = {
    "oxygen": "DO",
    "dissolved oxygen": "DO",
    "o2": "DO",
    "potential of hydrogen": "pH",
    "ammonia": "NH3",
    "ammonia nitrogen": "NH3",
    "temp": "temperature",
    "c": "temperature"
}

def resolve_entity(entity_string):
    """Maps raw extracted entities to their canonical database keys."""
    if not entity_string:
        return entity_string
    clean_str = entity_string.lower().strip()
    return CANONICAL_PARAMS.get(clean_str, entity_string)

def normalize_vernacular_input(text):
    if not text:
        return ""
    return text.strip()

def extract_semantic_triplets(query_lower, pending_metric=None, previous_subject=None):
    """
    Deterministic Triplet Parser with Explicit Slot Filling and Subject Inheritance
    """
    # 1. Detect Subject
    target_subjects = ["shrimp", "రొయ్యలు", "చేపలు", "fish"]
    subject = "shrimp/fish" if any(term in query_lower for term in target_subjects) else "pond/water"
    
    # 2. Extract Symptoms (Proximity mapping)
    is_negative = any(neg in query_lower for neg in ["not", "no", "don't", "cant", "doesn't", "లేదు", "తక్కువగా"])
    symptoms = []
    
    if re.search(r'(not|no|don\'t|doesn\'t|stop|lack).{0,15}eat', query_lower) or "తినడం లేదు" in query_lower:
        symptoms.append("not eating")
    elif "eat" in query_lower or "eating" in query_lower or "తినడం" in query_lower:
        symptoms.append("eating normally")
        
    if re.search(r'(not|no|don\'t|doesn\'t|stop).{0,15}swim', query_lower):
        symptoms.append("not swimming")
    elif "surface" in query_lower or "gasp" in query_lower or "పైకి" in query_lower:
        symptoms.append("surface gasping / swimming near surface")
        
    # FIX: Use word boundaries (\b) so "greenhouse" doesn't trigger "green"
    if re.search(r'\bgreen\b', query_lower) or re.search(r'\bcolor\b', query_lower) or "ఆకుపచ్చ" in query_lower:
        symptoms.append("dark green water")
        
    if "dying" in query_lower or "dead" in query_lower or "చనిపోతున్నాయి" in query_lower:
        symptoms.append("mortality / dying")
        
    # 3. Extract Numerical Values
    param_triplets = []
    
    # Try explicit matches first (e.g., "DO is 2.0")
    parameter_matches = re.findall(r'(oxygen|do|ph|temp|temperature).{0,30}?(\d+(?:\.\d+)?)', query_lower)
    for param, val in parameter_matches:
        canonical_param = resolve_entity(param)
        param_triplets.append((canonical_param, "value", float(val)))
        
    # SLOT FILLING & INHERITANCE FIX
    if not param_triplets and pending_metric:
        naked_number_match = re.search(r'(\d+(?:\.\d+)?)', query_lower)
        if naked_number_match:
            val = float(naked_number_match.group(1))
            canonical_pending = resolve_entity(pending_metric)
            param_triplets.append((canonical_pending, "value", val))
            if previous_subject and previous_subject != "pond/water":
                subject = previous_subject
            
    return subject, symptoms, param_triplets

def extract_hypothesis(farmer_query, session_state=None):
    """
    State-Aware Extractor (Zero-Compute)
    """
    cleaned_query = normalize_vernacular_input(farmer_query)
    
    if session_state is None:
        session_state = {"symptoms": set(), "pending_metric": None, "last_subject": None}
        
    print(f"\n[DEBUG] Running Parser on: '{cleaned_query}' | Pending Slot: {session_state.get('pending_metric')}")
    
    subject, new_symptoms, param_triplets = extract_semantic_triplets(
        cleaned_query.lower(), 
        pending_metric=session_state.get("pending_metric"),
        previous_subject=session_state.get("last_subject")
    )
    
    # --- NEW: PASS-THROUGH GATE ---
    # If we found zero symptoms, zero parameters, and aren't waiting on a metric,
    # this is either an informational question or an out-of-domain query.
    # Return the raw query so the router can see the "What" keyword and process it correctly.
    if not new_symptoms and not param_triplets and not session_state.get("pending_metric"):
        print(f"[+] Final Hypothesis (Pass-Through): {cleaned_query}")
        return cleaned_query, session_state
    # ------------------------------
    
    # Accumulate symptoms across turns and update last known subject
    session_state["symptoms"].update(new_symptoms)
    session_state["last_subject"] = subject
    
    # Clear pending metric now that we found a number, or keep it if missing
    if param_triplets:
        session_state["pending_metric"] = None
        
    # Assemble structured hypothesis string for ChromaDB
    symptom_str = ", ".join(session_state["symptoms"]) if session_state["symptoms"] else "abnormal behavior"
    param_str = f" with measured parameters {param_triplets}" if param_triplets else ""
    
    hypothesis_sentence = f"The {subject} exhibits {symptom_str}{param_str}. Condition state is definitive."
    
    print(f"[+] Final Hypothesis: {hypothesis_sentence}")
    return hypothesis_sentence, session_state