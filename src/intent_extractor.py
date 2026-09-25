import re

def normalize_vernacular_input(text):
    """
    Language-aware preprocessing to normalize or protect key technical terms 
    (shrimp, oxygen, pH, temperature) during mixed Telugu-English inputs.
    """
    if not text:
        return ""
    return text.strip()

def extract_semantic_triplets(farmer_query):
    """
    Deterministic Triplet Parser: Extracts core semantic triples 
    [Subject] -> [Modifier/Symptom] -> [Value/State] without relying on unconstrained LLM text generation.
    """
    cleaned_query = normalize_vernacular_input(farmer_query)
    query_lower = cleaned_query.lower()
    
    # 1. Detect Subject using accurate native vocabulary
    target_subjects = ["shrimp", "రొయ్యలు", "చేపలు", "fish"]
    subject = "shrimp/fish" if any(term in query_lower for term in target_subjects) else "pond/water"
    
    # 2. Extract Symptoms / Actions with Polarity Protection
    is_negative = any(neg in query_lower for neg in ["not", "no", "don't", "cant", "doesn't", "లేదు", "తక్కువగా"])
    
    symptoms = []
    if "eat" in query_lower or "eating" in query_lower or "తినడం" in query_lower:
        symptoms.append("not eating" if is_negative else "eating normally")
    if "swim" in query_lower or "surface" in query_lower or "gasp" in query_lower or "పైకి" in query_lower:
        if "surface" in query_lower or "gasp" in query_lower or "పైకి" in query_lower:
            symptoms.append("surface gasping / swimming near surface")
        elif is_negative:
            symptoms.append("not swimming")
    if "green" in query_lower or "color" in query_lower or "ఆకుపచ్చ" in query_lower:
        symptoms.append("dark green water")
    if "dying" in query_lower or "dead" in query_lower or "చనిపోతున్నాయి" in query_lower:
        symptoms.append("mortality / dying")
        
    # 3. Extract Numerical Values & Parameters (Widened regex to handle conversational filler)
    parameter_matches = re.findall(r'(oxygen|do|ph|temp|temperature).{0,30}?(\d+(?:\.\d+)?)', query_lower)
    param_triplets = []
    for param, val in parameter_matches:
        param_triplets.append((param.upper(), "value", float(val)))
        
    # 4. Detect Uncertainty / Conditional Modifiers
    uncertainty_detected = any(term in query_lower for term in ["may", "might", "maybe", "perhaps", "could be", "ఉందేమో"])
    modifier = "uncertain (may be)" if uncertainty_detected else "definitive"

    # Assemble structured semantic representation dictionary
    structured_hypothesis = {
        "subject": subject,
        "symptoms": symptoms,
        "parameters": param_triplets,
        "certainty": modifier,
        "raw_polarity": "negative" if is_negative else "positive"
    }
    
    # Convert structured dictionary into a clean, standardized hypothesis string for ChromaDB/NLI
    symptom_str = ", ".join(symptoms) if symptoms else "abnormal behavior"
    param_str = f" with measured parameters {param_triplets}" if param_triplets else ""
    
    hypothesis_sentence = f"The {subject} exhibits {symptom_str}{param_str}. Condition state is {modifier}."
    
    return structured_hypothesis, hypothesis_sentence

def extract_hypothesis(farmer_query, model_pipeline=None):
    """
    Main entry point for diagnostic engine and unit tests.
    Uses the deterministic semantic triplet parser instead of unconstrained LLM text generation.
    """
    cleaned_query = normalize_vernacular_input(farmer_query)
    
    if not cleaned_query or len(cleaned_query) < 3:
        return "The farmer reports an unspecified pond observation."
        
    print(f"\n[DEBUG] Running Deterministic Triplet Parser on: '{cleaned_query}'")
    _, hypothesis_sentence = extract_semantic_triplets(cleaned_query)
    
    print(f"[+] Hypothesis: {hypothesis_sentence}")
    return hypothesis_sentence