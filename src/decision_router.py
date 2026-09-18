import os
import torch
from sentence_transformers import CrossEncoder
from intent_extractor import extract_hypothesis
# Prevent thread contention on Mac/CPU hardware
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
torch.set_num_threads(4)

def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class NLIDecisionRouter:
    def __init__(self, ce_model_name="./models/deberta_aquaculture_router"):
        # Initialize the Cross-Encoder once when the router is created
        self.ce_model = CrossEncoder(ce_model_name)
        
    def get_routing_decision(self, entailment_score, neutral_score, contradiction_score):
        """
        Applies the deterministic thresholds defined in Phase 2C.
        """
        # 1. Biological danger / Contradiction (ABSTAIN)
        if contradiction_score > 0.30:
            return "ABSTAIN"
        
        # 2. Missing diagnostic info (CLARIFY)
        elif neutral_score > 0.40:
            return "CLARIFY"
            
        # 3. Safe and complete match (ANSWER)
        elif entailment_score >= 0.75:
            return "ANSWER"
            
        # Failsafe fallback
        return "CLARIFY"

    def route_query(self, llm_instance, raw_query, retrieved_chunk):
        """
        The full Phase 2 pipeline: Extract -> NLI Compare -> Route
        """
        # 1. Flatten the messy input using your Qwen 0.5B instance
        clean_hypothesis = extract_hypothesis(llm_instance, raw_query)
        
        # 2. Compare the clean hypothesis against the retrieved ChromaDB chunk
        scores = self.ce_model.predict([(retrieved_chunk, clean_hypothesis)])[0]
        
        # DeBERTa-v3 outputs logits for [Contradiction, Entailment, Neutral]
        contradiction_score = scores[0]
        entailment_score = scores[1]
        neutral_score = scores[2]
        
        # 3. Pass scores through the logic gates
        decision = self.get_routing_decision(entailment_score, neutral_score, contradiction_score)
        
        return {
            "decision": decision,
            "extracted_hypothesis": clean_hypothesis,
            "scores": {
                "entailment": float(entailment_score),
                "neutral": float(neutral_score),
                "contradiction": float(contradiction_score)
            }
        }
    
    def evaluate_routing(self, extracted_hypothesis, retrieved_chunk):
        """
        Compares the hypothesis to the retrieved chunk and returns the decision and metrics.
        Used by the ablation test block and Phase 2 modular testing.
        """
        # CRITICAL: apply_softmax=True converts raw logits into 0.0-1.0 probabilities 
        # so your 0.75, 0.40, and 0.30 thresholds mathematically function.
        scores = self.ce_model.predict([(retrieved_chunk, extracted_hypothesis)], apply_softmax=True)[0]
        
        # DeBERTa-v3 cross-encoder label mapping: [Contradiction, Entailment, Neutral]
        contradiction_score = scores[0]
        entailment_score = scores[1]
        neutral_score = scores[2]
        
        # Pass through your deterministic gates
        decision = self.get_routing_decision(entailment_score, neutral_score, contradiction_score)
        
        metrics = {
            "entailment": float(entailment_score),
            "neutral": float(neutral_score),
            "contradiction": float(contradiction_score)
        }
        
        return decision, metrics


# --- Pipeline Logic Test ---
if __name__ == "__main__":
    router = NLIDecisionRouter()
    
    # 1. Simulating the farmer's raw input
    raw_farmer_input = "The water turned dark green yesterday, shrimp aren't eating, and I see white spots."
    print(f"\n[1] Raw Input: {raw_farmer_input}")
    
    # 2. Simulating Qwen's Stage-2 Extraction output
    extracted_hypothesis = "The pond exhibits a phytoplankton bloom, loss of appetite, and suspected white spots on shrimp."
    print(f"[2] Extracted Hypothesis: {extracted_hypothesis}")
    
    # 3. Simulating Stage-3 Retrieved Chunks (BGE-M3)
    context_match = "For ponds exhibiting phytoplankton blooms and loss of appetite with suspected White Spot Syndrome Virus (WSSV), immediate quarantine and zero water exchange is required."
    context_contradict = "To treat a phytoplankton bloom and white spots, increase the feeding rate heavily and raise pH to 10.0."
    context_neutral = "Phytoplankton blooms can cause fluctuations in dissolved oxygen. Monitor the water quality regularly."

    # 4. Testing the Stage-4 Gate
    print("\n--- Gate Testing ---")
    dec, met = router.evaluate_routing(extracted_hypothesis, context_match)
    print(f"Match Context      -> Decision: {dec} | Scores: {met}")

    dec, met = router.evaluate_routing(extracted_hypothesis, context_contradict)
    print(f"Contradict Context -> Decision: {dec} | Scores: {met}")

    dec, met = router.evaluate_routing(extracted_hypothesis, context_neutral)
    print(f"Neutral Context    -> Decision: {dec} | Scores: {met}")