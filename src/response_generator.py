import re

def generate_answer(hypothesis, evidence_list):
    """
    Extracts the pre-computed English/Telugu advice from the top evidence chunk
    and explicitly grounds the numerical parameters so the final answer does not drift.
    """
    if not evidence_list:
        return {
            "english": "No conclusive evidence found in the local database.",
            "telugu": "స్థానిక డేటాబేస్‌లో ఎలాంటి ఆధారాలు కనుగొనబడలేదు."
        }
        
    top_chunk = evidence_list[0]
    
   # Safely extract from the nested 'metadata' dictionary
    chunk_metadata = top_chunk.get("metadata", {})
    base_english = chunk_metadata.get("english_synthesis", "No advice available.")
    base_telugu = chunk_metadata.get("telugu_translation", "సలహా అందుబాటులో లేదు.")
    
    # --- GROUNDING DRIFT FIX ---
    # Search the hypothesis for parameters like [('OXYGEN/DO', 'value', 2.0)]
    param_match = re.search(r"\('([^']+)',\s*'value',\s*([\d\.]+)\)", hypothesis)
    
    if param_match:
        metric = param_match.group(1)
        val = param_match.group(2)
        
        # Inject the farmer's specific metric at the start of the retrieved advice
        grounded_english = f"**[Alert: {metric} at {val}]** {base_english}"
        grounded_telugu = f"**[హెచ్చరిక: {metric} స్థాయి {val} గా నమోదైంది]** {base_telugu}"
        
        return {
            "english": grounded_english,
            "telugu": grounded_telugu
        }

    # If no numbers were involved, return the base database strings
    return {
        "english": base_english,
        "telugu": base_telugu
    }