import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Hardware optimization
device = "mps" if torch.backends.mps.is_available() else "cpu"
SRC_MODEL = "facebook/nllb-200-distilled-600M"

def load_translator():
    print(f"[*] Loading NLLB-200 on {device.upper()} in FP16...")
    tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)
    
    # FP16 (Half-Precision) to maintain high-speed batch processing
    model = AutoModelForSeq2SeqLM.from_pretrained(
        SRC_MODEL,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    
    return tokenizer, model

def translate_to_telugu(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(device)
    
    # NLLB language code for Telugu
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("tel_Telu")
    
    with torch.no_grad():
        # Using greedy decoding (num_beams=1) for speed
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=1
        )
        
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translation.strip()

if __name__ == "__main__":
    # This is the successful output from Task 1.3
    synthesized_advice = "If oxygen levels drop to 3 mg/L, shrimp mortality increases."
    
    print(f"[*] Input English: '{synthesized_advice}'")
    tokenizer, model = load_translator()
    
    telugu_translation = translate_to_telugu(synthesized_advice, tokenizer, model)
    
    print("\n[Task 1.4 Output]")
    print(f"Telugu: {telugu_translation}")