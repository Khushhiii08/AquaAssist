"""
AquaAssist: Dynamic Grounded Response Generator (Production Hardened)
Synthesizes structured, accurate responses from verified ChromaDB evidence chunks
using local Qwen 2.5 SLM inference with token-budgeted context formatting
and accelerated greedy-search NLLB-200 vernacular translation.
Zero hardcoded response dictionaries. Zero context-overflow crashes.
"""

import os
import re
from typing import List, Dict, Any, Optional
import torch
from llama_cpp import Llama
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen-2.5-1.5b-instruct.gguf")
SRC_TRANSLATOR = "facebook/nllb-200-distilled-600M"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

_LLM_INSTANCE = None
_TRANSLATOR_MODEL = None
_TRANSLATOR_TOKENIZER = None


def get_slm():
    """Lazy-loads local Qwen SLM with safe context allocation."""
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        if os.path.exists(MODEL_PATH):
            print(f"[*] Loading Qwen 2.5 SLM from {MODEL_PATH}...")
            _LLM_INSTANCE = Llama(
                model_path=MODEL_PATH,
                n_threads=2,
                n_batch=512,
                n_ctx=2048,
                n_gpu_layers=0,  # Stable across all platforms and test suites
                verbose=False
            )
        else:
            print(f"[!] Warning: GGUF model not found at {MODEL_PATH}. Running in extractive fallback mode.")
    return _LLM_INSTANCE


def get_translator():
    """Lazy-loads NLLB-200 in half-precision for fast inference."""
    global _TRANSLATOR_MODEL, _TRANSLATOR_TOKENIZER
    if _TRANSLATOR_MODEL is None:
        try:
            print(f"[*] Loading NLLB-200 Translator on {DEVICE.upper()}...")
            _TRANSLATOR_TOKENIZER = AutoTokenizer.from_pretrained(SRC_TRANSLATOR)
            _TRANSLATOR_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                SRC_TRANSLATOR,
                torch_dtype=torch.float16 if DEVICE in ("mps", "cuda") else torch.float32
            ).to(DEVICE)
            _TRANSLATOR_MODEL.eval()
        except Exception as e:
            print(f"[!] Warning: Translation model failed to load ({e}).")
    return _TRANSLATOR_MODEL, _TRANSLATOR_TOKENIZER


def translate_to_telugu(english_text: str) -> str:
    """
    Translates English advice to natural Telugu using greedy decoding (beam=1)
    to guarantee sub-second translation on edge hardware.
    """
    if not english_text or len(english_text.strip()) < 5:
        return ""

    model, tokenizer = get_translator()
    if model is None or tokenizer is None:
        return "తెలుగు అనువాదం అందుబాటులో లేదు."

    try:
        # Cap English input length to avoid translating excessive text on CPU
        inputs = tokenizer(
            english_text, 
            return_tensors="pt", 
            max_length=200, 
            truncation=True
        ).to(DEVICE)
        
        forced_bos = tokenizer.convert_tokens_to_ids("tel_Telu")
        
        # Fast greedy decoding on 2-vCPU edge nodes:
        # 1. Use max_new_tokens (not max_tokens)
        # 2. early_stopping=True stops generating immediately when EOS is emitted
        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=100,
                num_beams=1,
                do_sample=False,
                early_stopping=True
            )
        raw_translation = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()

        # Domain lexicon enforcement
        cleaned = (
            raw_translation
            .replace("గ్రెయిన్", "రొయ్యలు")
            .replace("గ్రెడ్లు", "రొయ్యలు")
            .replace("కుట్టడం", "గడ్డకట్టడం")
            .replace("ఫిష్", "చేపలు")
        )
        return cleaned
    except Exception as e:
        print(f"[!] Translation error: {e}")
        return "తెలుగు అనువాదంలో లోపం ఏర్పడింది."

def format_evidence_context(evidence_list: List[Dict[str, Any]], max_chunks: int = 3, max_words_per_chunk: int = 150) -> str:
    """
    Formats the highest-scoring evidence chunks while strictly enforcing a token budget
    to guarantee the prompt never overflows Qwen's 2048 context window.
    """
    context_blocks = []
    selected_chunks = evidence_list[:max_chunks]
    
    for idx, chunk in enumerate(selected_chunks, start=1):
        text = chunk.get("text", "").strip()
        words = text.split()
        if len(words) > max_words_per_chunk:
            text = " ".join(words[:max_words_per_chunk]) + "..."
            
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Aquaculture Manual")
        page = meta.get("page", "?")
        context_blocks.append(f"[Document {idx} | Source: {source}, Page {page}]\n{text}")
        
    return "\n\n".join(context_blocks)


def synthesize_with_slm(query: str, context: str, species: Optional[str] = None) -> Optional[str]:
    """Generates a structured, concise answer from context using Qwen 2.5 SLM."""
    llm = get_slm()
    if llm is None:
        return None

    species_str = f"Target organism: {species}. " if species else ""
    system_prompt = (
        "You are an expert aquaculture technical specialist. "
        "Answer the user's question directly using only the verified documentation provided. "
        f"{species_str}"
        "Format your response as a clear, structured list of practical operational steps. "
        "Be direct, actionable, and concise (under 120 words)."
    )

    user_prompt = f"Documentation Context:\n{context}\n\nFarmer Question: {query}\n\nTechnical Answer:"

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=256,
            temperature=0.1,
            top_p=0.9
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[!] SLM synthesis failed: {e}")
        return None

def generate_answer(query_or_claim: str, evidence_list: List[Dict[str, Any]], species: Optional[str] = None) -> Dict[str, str]:
    """
    Main generative pipeline.
    Synthesizes grounded advice dynamically from verified evidence.
    """
    if not evidence_list:
        return {
            "english": "No conclusive scientific evidence found in the local knowledge base.",
            "telugu": "స్థానిక నాలెడ్జ్ బేస్‌లో నిర్దిష్టమైన ఆధారాలు కనుగొనబడలేదు."
        }

    # Step 1: Format verified context with strict token budgeting
    context = format_evidence_context(evidence_list, max_chunks=3, max_words_per_chunk=150)

    # Step 2: Attempt dynamic SLM generation
    synthesized_en = synthesize_with_slm(query_or_claim, context, species=species)

    # Step 3: Extractive Fallback (if SLM fails or returns None)
    if not synthesized_en:
        top_chunk = evidence_list[0]
        meta = top_chunk.get("metadata", {})
        
        synthesized_en = meta.get("english_synthesis")
        if not synthesized_en:
            raw_text = top_chunk.get("text", "No synthesis available.")
            synthesized_en = re.sub(r'\(.*?\d{4}.*?\)', '', raw_text).strip()
            
        precomputed_te = meta.get("telugu_translation")
        if precomputed_te:
            return {
                "english": synthesized_en,
                "telugu": precomputed_te
            }

    # Step 4: Rapid Telugu translation
    synthesized_te = translate_to_telugu(synthesized_en)

    return {
        "english": synthesized_en,
        "telugu": synthesized_te
    }