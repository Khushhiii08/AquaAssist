"""
AquaAssist: Multi-Tier Intent & Safety Policy Router (Production Hardened)
Evaluates incoming queries through 4 deterministic gates:
1. Tier 1: Broad Aquaculture Domain Filter (via spaCy lemma matching)
2. Tier 2: Chemical Safety & Hazardous Mixture Intercept (T8, T9)
3. Tier 3: Temporal & Commercial Market Data Gate (T7)
4. Tier 4: Vague Diagnostic Query Intercept (T6)
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
    "copper", "sulfate", "paracetamol", "antibiotic",
    "మేత", "సున్నం", "మందు"
}

# Explicit clinical signs
AQUACULTURE_CLINICAL_LEMMAS = {
    "disease", "mortality", "die", "dead", "dying", "gasp", "gasping", "suffocate",
    "sluggish", "erratic", "lesion", "ulcer", "rot", "spot", "whiteness", "necrosis",
    "చనిపోతున్నాయి", "మరణాలు", "కుళ్ళు"
}

# Normalization for multi-word chemical compounds
CHEMICAL_PATTERNS = {
    "bleach": [r"\bbleach\b", r"\bbleaching powder\b", r"\bhypochlorite\b", r"\bchlorine\b"],
    "permanganate": [r"\bpotassium permanganate\b", r"\bpermanganate\b", r"\bkmno4\b"],
    "formalin": [r"\bformalin\b", r"\bformaldehyde\b"],
    "copper_sulfate": [r"\bcopper sulfate\b", r"\bcopper sulphate\b", r"\bcoppersulfate\b", r"\bblue vitriol\b"],
    "ammonia": [r"\bammonia\b", r"\bammonium\b"]
}

# Dangerous combinations: (Chemical A, Chemical B)
HAZARDOUS_PAIRS = [
    ("bleach", "permanganate"),
    ("bleach", "formalin"),
    ("bleach", "ammonia"),
    ("formalin", "permanganate"),
    ("copper_sulfate", "formalin")  # Lethal dual biocide / acute DO collapse (T9)
]


def is_aquaculture_domain(doc: spacy.tokens.Doc) -> bool:
    """Tier 1 Gate: Confirms the inquiry belongs strictly to aquaculture."""
    text_lower = doc.text.lower()
    
    # Check terrestrial exclusion markers
    terrestrial_markers = ["wheat", "paddy", "rice", "cotton", "sugarcane", "maize", "crop", "tractor", "potash"]
    if any(re.search(rf"\b{m}\b", text_lower) for m in terrestrial_markers):
        # Allow if pond/aquaculture is explicitly co-present
        if not any(aq in text_lower for aq in ["pond", "fish", "shrimp", "aquaculture"]):
            return False

    lemmas = {token.lemma_.lower() for token in doc}
    words = {token.text.lower() for token in doc}
    all_tokens = lemmas | words

    if bool(all_tokens & AQUACULTURE_DOMAIN_LEMMAS) or bool(all_tokens & AQUACULTURE_CLINICAL_LEMMAS):
        return True

    telugu_markers = ["చేప", "రొయ్య", "చెరువు", "ఆక్సిజన్", "మేత", "చనిపో", "నీరు", "మందు"]
    return any(m in text_lower for m in telugu_markers)


def detect_chemical_hazards(doc: spacy.tokens.Doc) -> Optional[str]:
    """Tier 2 Gate: Detects dangerous chemical combinations and blind mortality dosing."""
    text_lower = doc.text.lower()
    lemmas = {token.lemma_.lower() for token in doc}
    all_tokens = lemmas | {token.text.lower() for token in doc}

    # 1. Detect which chemical families are present via regex
    detected_chemicals = set()
    for chem_key, patterns in CHEMICAL_PATTERNS.items():
        if any(re.search(pat, text_lower) for pat in patterns):
            detected_chemicals.add(chem_key)

    # Check for hazardous mixing pairs
    for chem_a, chem_b in HAZARDOUS_PAIRS:
        if chem_a in detected_chemicals and chem_b in detected_chemicals:
            # Check for combination intent
            mix_markers = ["mix", "combine", "blend", "add", "together", "with", "and", "along with", "కలపడం"]
            if any(m in text_lower for m in mix_markers):
                return "CHEMICAL_HAZARD"

    # 2. Blind Chemical Dosing for Acute Mortality (T6)
    dose_triggers = ["how many kg", "how much", "what dose", "dosage", "how much powder", "how many grams", "మోతాదు"]
    has_dose_inquiry = any(dt in text_lower for dt in dose_triggers)

    has_toxic_chemical = bool(detected_chemicals) or any(c in all_tokens for c in {"chemical", "medicine", "powder"})
    has_mortality = any(m in all_tokens for m in {"dying", "dead", "mortality", "losses", "die", "kill", "చనిపోతున్నాయి"})

    if has_dose_inquiry and has_toxic_chemical and has_mortality:
        return "BLIND_DOSING_REFUSAL"

    return None


def detect_temporal_or_market_query(doc: spacy.tokens.Doc) -> bool:
    """Tier 3 Gate: Intercepts temporal commodity/market requests."""
    text_lower = doc.text.lower()
    market_keywords = ["market price", "price today", "rate per kg", "cost per kg", "rate today", "bhimavaram today", "mandi price", "ధర"]
    return any(kw in text_lower for kw in market_keywords)


def is_vague_complaint(doc: spacy.tokens.Doc) -> bool:
    """
    Tier 4 Gate: Intercepts ambiguous complaints lacking clinical symptoms or water quality metrics.
    """
    text_lower = doc.text.lower().strip()
    
    # Emotive distress / vague quality descriptors
    vague_descriptors = ["bad", "strange", "weird", "dirty", "unusual", "not good", "problem"]
    has_vague_descriptor = any(vd in text_lower for vd in vague_descriptors)
    
    # Check for concrete diagnostic observables
    has_telemetry = any(t.like_num or t.pos_ == "NUM" for t in doc)
    has_clinical_symptoms = any(
        s in text_lower for s in [
            "gasping", "mortality", "dead", "dying", "white spot", "red spot", "lesion",
            "rot", "smell", "odor", "floating", "sluggish", "erratic", "green", "turbid", "surface"
        ]
    )

    # If the user complains that water/pond is "bad/strange" with zero observables
    if ("pond" in text_lower or "water" in text_lower) and has_vague_descriptor:
        if not has_telemetry and not has_clinical_symptoms:
            return True

    return False


def route_query_policy(raw_query: str) -> Tuple[str, Optional[str]]:
    """
    Primary routing decision engine.
    Returns: (route_status, override_message_key)
    """
    q_clean = raw_query.strip()
    doc = nlp(q_clean)

    # Tier 1: Aquaculture Domain Gate
    if not is_aquaculture_domain(doc):
        return "POLICY_INTERCEPT", "OUT_OF_DOMAIN"

    # Tier 2: Chemical Safety & Hazard Intercept (T2, T6, T9)
    hazard_key = detect_chemical_hazards(doc)
    if hazard_key:
        return "POLICY_INTERCEPT", hazard_key

    # Tier 3: Temporal / Real-time Commodity Limitation (T8)
    if detect_temporal_or_market_query(doc):
        return "POLICY_INTERCEPT", "REALTIME_DATA_LIMITATION"

    # Tier 4: Vague Diagnostic Gate (T10)
    if is_vague_complaint(doc):
        return "POLICY_INTERCEPT", "VAGUE_DIAGNOSTIC"

    return "PASS", None


def evaluate_policy_tiers(query: str, session_state: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """Compatibility wrapper adapting route_query_policy to benchmark suite."""
    route_status, key = route_query_policy(query)
    if route_status == "POLICY_INTERCEPT":
        return True, key, None
    return False, None, None