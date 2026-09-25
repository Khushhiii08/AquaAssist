import os

def generate_answer(hypothesis, evidence_chunks):
    """
    Task 2.5: Metadata Unpacking.
    Extracts the pre-computed LLM synthesis and NMT translation directly 
    from ChromaDB metadata, resulting in zero runtime generative latency.
    """
    if not evidence_chunks:
        return {
            "english": "No relevant evidence found in the offline database.",
            "telugu": "ఆఫ్‌లైన్ డేటాబేస్‌లో సంబంధిత సమాచారం కనుగొనబడలేదు."
        }
        
    # Extract the highest-confidence chunk
    best_chunk = evidence_chunks[0]
    metadata = best_chunk.get("metadata", {})
    
    # Pull the exact conversational strings we compiled during Ingestion
    english_ui = metadata.get("english_synthesis", "Observation noted. Please consult a specialist.")
    telugu_ui = metadata.get("telugu_translation", "గమనిక నమోదు చేయబడింది. దయచేసి నిపుణుడిని సంప్రదించండి.")
    
    return {
        "english": english_ui,
        "telugu": telugu_ui
    }