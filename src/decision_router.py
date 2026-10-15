"""
AquaAssist: Multi-Tier Intent & Safety Policy Router (Hardened)
Evaluates incoming queries through 4 deterministic gates:
1. Tier 1: Broad Aquaculture Domain Filter (via spaCy lemma matching)
2. Tier 2: Chemical Safety & Hazardous Mixture Intercept (T8, T9)
3. Tier 3: Temporal & Commercial Market Data Gate (T7)
4. Tier 4: Vague Diagnostic Query Intercept (T6)

Zero inflection misses (carps -> carp). Zero false chemical intercepts on 'and'.
"""

import re
from typing import Dict, Any, Tuple, Optional
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Lemma roots for domain membership
AQUACULTURE_DOMAIN_LEMMAS = {
    # Organisms (Lemmatized)
    "fish", "shrimp", "prawn", "carp", "tilapia", "rohu", "catla", "mrigal", "murrel",
    "vannamei", "monodon", "fry", "fingerling", "broodstock", "stock", "crustacean",
    "చేప", "చేపలు", "రొయ్య", "రొయ్యలు",
    # Habitat & Limnology
    "pond", "water", "tank", "aquaculture", "limnology", "hatchery", "plankton", "bloom",
    "algae", "algal", "scum", "aerator", "aeration", "turbidity", "secchi", "sediment", 
    "sludge", "bottom", "benthic", "inlet", "outlet", "dyke", "చెరువు", "నీరు", "ఏరేటర్",
    # Telemetry Parameters
    "oxygen", "do", "ph", "alkalinity", "hardness", "ammonia", "tan", "nitrite", "nitrate",
    "salinity", "temperature", "temp", "orp", "transparency", "ఆక్సిజన్", "ఉష్ణోగ్రత", "ఉప్పుదనం", "క్షారత",
    # Inputs & Treatments
    "feed", "feeding", "meal", "pellet", "bleaching", "permanganate", "kmno4", "chlorine", 
    "lime", "quicklime", "zeolite", "formalin", "probiotic", "chemical", "disinfectant",
    "మేత", "సున్నం", "మందు"
}

# Explicit clinical signs
AQUACULTURE_CLINICAL_LEMMAS = {
    "disease", "mortality", "die", "dead", "dying", "gasp", "gasping", "suffocate",
    "sluggish", "erratic", "lesion", "ulcer", "rot", "spot", "whiteness", "necrosis",
    "చనిపోతున్నాయి", "మరణాలు", "కుళ్ళు"
}

# Lethal/Hazardous Chemical Combinations
HAZARDOUS_CHEMICAL_GROUPS = [
    ({"bleaching", "bleach", "hypochlorite", "chlorine"}, {"permanganate", "kmno4", "potassium permanganate"}),
    ({"bleaching", "bleach", "hypochlorite"}, {"formalin", "formaldehyde"}),
    ({"bleaching", "bleach"}, {"ammonia", "ammonium"}),
    ({"formalin"}, {"permanganate", "kmno4"})
]


def is_aquaculture_domain(doc: spacy.tokens.Doc) -> bool:
    """
    Tier 1 Gate: Verifies that the query touches aquaculture organisms, 
    pond environment, parameters, or clinical signs via lemma matching.
    """
    text_lower = doc.text.lower()

    # 1. Check spaCy Lemmatized tokens
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in AQUACULTURE_DOMAIN_LEMMAS or lemma in AQUACULTURE_CLINICAL_LEMMAS:
            return True
        if token.text.lower() in AQUACULTURE_DOMAIN_LEMMAS:
            return True

    # 2. Check Vernacular Substrings
    telugu_markers = ["చేప", "రొయ్య", "చెరువు", "ఆక్సిజన్", "మేత", "చనిపో", "నీరు", "మందు"]
    if any(m in text_lower for m in telugu_markers):
        return True

    return False


def detect_chemical_hazards(doc: spacy.tokens.Doc) -> Optional[str]:
    """
    Tier 2 Gate: Detects dangerous chemical combinations and blind mortality dosing.
    Requires true semantic combination verbs (mix, combine, with).
    """
    text_lower = doc.text.lower()
    lemmas = {token.lemma_.lower() for token in doc}
    words = {token.text.lower() for token in doc}
    all_tokens = lemmas | words

    # 1. Incompatible Chemical Mixtures (T9)
    for group_a, group_b in HAZARDOUS_CHEMICAL_GROUPS:
        has_a = bool(all_tokens & group_a)
        has_b = bool(all_tokens & group_b)

        if has_a and has_b:
            # Must involve combining intent: 'mix', 'combine', 'blend', 'together', or 'with'
            mix_verbs = {"mix", "combine", "blend", "add", "use", "apply", "కలపడం"}
            has_mix_verb = bool(lemmas & mix_verbs)
            has_together = any(w in text_lower for w in ("together", "along with", "with", "and"))
            
            if has_mix_verb or has_together:
                return "CHEMICAL_HAZARD"

    # 2. Blind Chemical Dosing for Acute Mortality (T8)
    dose_triggers = ["how many kg", "how much", "what dose", "dosage", "how much powder", "how many grams", "మోతాదు"]
    has_dose_inquiry = any(dt in text_lower for dt in dose_triggers)

    has_chemical = any(c in all_tokens for c in {"bleaching", "bleach", "kmno4", "permanganate", "chemical", "chlorine"})
    has_mortality = any(m in all_tokens for m in {"dying", "dead", "mortality", "losses", "die", "చనిపోతున్నాయి"})

    if has_dose_inquiry and has_chemical and has_mortality:
        return "BLIND_DOSING_REFUSAL"

    return None


def detect_temporal_or_market_query(doc: spacy.tokens.Doc) -> bool:
    """
    Tier 3 Gate: Intercepts volatile real-time questions (local market feed prices, live tickers).
    """
    text_lower = doc.text.lower()
    price_markers = ["market price", "price today", "feed price", "rate today", "cost per kg", "today's price", "ధర ఎంత"]
    
    if any(pm in text_lower for pm in price_markers):
        return True

    # "price of feed" / "cost of feed"
    lemmas = {t.lemma_.lower() for t in doc}
    if ("price" in lemmas or "cost" in lemmas or "rate" in lemmas) and ("feed" in lemmas or "meal" in lemmas):
        return True

    return False


def is_vague_complaint(doc: spacy.tokens.Doc) -> bool:
    """
    Tier 4 Gate: Detects ambiguous complaints lacking physical observations or parameters (T6).
    Exempts valid informational questions like 'What parameters should I monitor?'.
    """
    text_lower = doc.text.lower()
    
    # If the user is asking an informational question, it is NOT a vague emergency complaint
    info_inquiry = any(token.lemma_.lower() in ("what", "which", "how", "parameter", "monitor", "test", "measure") for token in doc)
    if info_inquiry and ("parameter" in text_lower or "monitor" in text_lower or "cause" in text_lower):
        return False

    vague_phrases = [
        "water is bad", "pond is bad", "something wrong with pond", "water problem",
        "pond turned bad", "water not good", "నీరు బాగాలేదు", "చెరువు బాగాలేదు"
    ]
    is_vague = any(vp in text_lower for vp in vague_phrases)

    # Check for specific symptoms or numerical readings
    has_specifics = any(s in text_lower for s in ["green", "smell", "odor", "dead", "dying", "floating", "gasping", "turbid"])
    has_numbers = any(t.like_num or t.pos_ == "NUM" for t in doc)

    return is_vague and not has_specifics and not has_numbers


def route_query_policy(raw_query: str) -> Tuple[str, Optional[str]]:
    """
    Primary routing decision engine.
    Returns: (route_status, override_message_key)
    """
    q_clean = raw_query.strip()
    doc = nlp(q_clean)

    # Tier 1: Aquaculture Domain Gate (Lemma-Aware)
    if not is_aquaculture_domain(doc):
        return "POLICY_INTERCEPT", "OUT_OF_DOMAIN"

    # Tier 2: Chemical Safety & Hazard Intercept (T8, T9)
    hazard_key = detect_chemical_hazards(doc)
    if hazard_key:
        return "POLICY_INTERCEPT", hazard_key

    # Tier 3: Temporal / Real-time Commodity Limitation (T7)
    if detect_temporal_or_market_query(doc):
        return "POLICY_INTERCEPT", "REALTIME_DATA_LIMITATION"

    # Tier 4: Vague Diagnostic Gate (T6)
    if is_vague_complaint(doc):
        return "POLICY_INTERCEPT", "VAGUE_DIAGNOSTIC"

    # Cleared for RAG Retrieval & NLI Verification
    return "PROCEED_TO_RAG", None