# Meghana, you need to import extract_hypothesis() function to wire it into the main llama.cpp pipeline.
# The below strict zero-shot extractor prompt is designed for Qwen-2.5-0.5B, which is a smaller model suitable for running on a single GPU or even CPU for testing purposes.
# It will force Qwen to compress farmer's messy paragraphs into flat, declarative statements that can be used for NLI evaluation in the next stage of the pipeline.

# The strict zero-shot system prompt designed for Qwen-2.5-0.5B
EXTRACTOR_SYSTEM_PROMPT = """You are a strict data extraction system for aquaculture diagnostics. 
Your ONLY task is to convert conversational, messy descriptions of pond conditions into a single, clean declarative factual statement.
Do not answer the user's question. Do not provide advice. Do not include conversational filler.

RULES:
1. Identify the core biological or chemical symptoms described by the farmer.
2. Translate the intent into exactly one sentence starting with "The pond exhibits..."
3. Remove all questions, greetings, and requests for help.
4. Output NOTHING except the final declarative hypothesis.
"""

def build_extraction_prompt(messy_query: str) -> list:
    """
    Formats the input into Qwen's ChatML structure for the 0.5B model.
    Meghana will pass this directly into the llama.cpp instance.
    """
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": messy_query}
    ]
    return messages

def extract_hypothesis(llm_instance, messy_query: str) -> str:
    """
    Executes the Pre-Retrieval Extraction using the configured SLM.
    """
    messages = build_extraction_prompt(messy_query)
    
    # Qwen-2.5-0.5B inference call via llama-cpp-python
    response = llm_instance.create_chat_completion(
        messages=messages,
        max_tokens=64, # Hard limit to prevent runaway generation
        temperature=0.1, # Keep it highly deterministic
        stop=["\n", "User:", "<|im_end|>"]
    )
    
    # Clean and return the flattened declarative statement
    clean_hypothesis = response["choices"][0]["message"]["content"].strip()
    return clean_hypothesis

# --- Quick Local Test ---
if __name__ == "__main__":
    from llama_cpp import Llama
    
    print("[*] Loading Qwen-2.5-0.5B for Extraction Test...")
    
    # Initialize the lightweight model for Phase 2A
    extractor_llm = Llama(
        model_path="./models/qwen-2.5-0.5b-instruct.gguf",
        n_ctx=512,
        n_threads=4,
        verbose=False
    )
    
    # Test cases simulating the messy colloquial input from farmers
    test_cases = [
        "My water turned dark green yesterday, shrimp aren't eating, and I see white spots on their shells. What medicine should I buy?",
        "The pH reading is 9.8 today and the fish are swimming at the top gasping for air.",
        "Yellow floating matter in the tank and it smells like ammonia."
    ]
    
    for i, test_query in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Messy Input: {test_query}")
        
        hypothesis = extract_hypothesis(extractor_llm, test_query)
        
        print(f"Flattened Hypothesis: {hypothesis}")