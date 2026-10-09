import ollama

# We will test with a valid biological triplet and the noisy statistical triplet you just generated
sample_triplets = [
    "Observation: shrimp exhibit mortality. Context: The shrimp exhibit mortality when oxygen falls below 3 mg/L.",
    "Observation: analyses performed using involving Excel matrix ver. Context: Graphical analyses were performed using Microsoft Excel and the statistical determination involving ANOVA and correlation matrix in SPSS software ver."
]

# Set this to whichever model you downloaded via Ollama
OLLAMA_MODEL = "qwen2.5:1.5b" 

def synthesize_advice(triplet_chunk):
    """
    Task 1.3: Converts raw scientific triplets into conversational farmer advice,
    while aggressively filtering out academic/statistical noise.
    """
    system_prompt = "You are an expert aquaculture assistant. Be concise, empathetic, and strictly factual."
    
    user_prompt = f"""Read the following observation and context extracted from a scientific aquaculture manual. 

If the text contains actionable advice, biological facts, or environmental warnings for a farmer, rewrite it into a short, simple, 1-2 sentence response. Use a helpful, conversational tone.
If the text is just administrative, statistical, or academic noise (like software names, data analysis methods, or document formatting), output exactly and only the word: [SKIP]

Data:
{triplet_chunk}
"""
    
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        return f"[!] Error calling Ollama: {e}\n(Make sure the Ollama app is running and the model is pulled)"

if __name__ == "__main__":
    print(f"[*] Testing Task 1.3: Local LLM Synthesis via Ollama ({OLLAMA_MODEL})")
    
    for i, chunk in enumerate(sample_triplets):
        print(f"\n--- Test Sample {i+1} ---")
        print(f"INPUT: {chunk}")
        synthesis = synthesize_advice(chunk)
        print(f"OUTPUT: {synthesis}")