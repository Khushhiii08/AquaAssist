"""
AquaAssist: Multi-Tier Intent & Safety Policy Router (Production Hardened)
Evaluates incoming queries through 4 deterministic gates:
1. Tier 1: Broad Aquaculture Domain Filter (spaCy lemma matching + Romanized & Telugu Unicode script + state awareness)
2. Tier 2: Chemical Safety, Therapeutic Dosing, Banned Substances, & Prescription Gates (T2, T6, T9)
3. Tier 3: Temporal & Commercial Market Data Gate (T8)
4. Tier 4: Vague Diagnostic & Missing Context Query Intercept (T10)
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
    # Organisms (Lemmatized English)
    "fish", "shrimp", "prawn", "carp", "tilapia", "rohu", "catla", "mrigal", "murrel",
    "vannamei", "monodon", "fry", "fingerling", "broodstock", "stock", "crustacean",
    "crab", "scampi", "seabass", "milkfish", "mullet", "seabream", "goldfish",
    # Organisms (Telugu Script)
    "చేప", "చేపలు", "రొయ్య", "రొయ్యలు", "పీత", "పీతలు",
    # Romanized Telugu transliterations
    "chepa", "chepalu", "royya", "royyalu", "peetha", "peethalu",
    "cheruvu", "cheruvulo", "neeru", "neellu", "vellu", "udayam", "sayantram",
    "entha", "veyali", "kalapali", "penchavacha", "vastunda", "taggindi", "perigindi",
    "mandhu", "sunnam", "theetha", "thindi", "meetha",
    # Habitat, Limnology & Units
    "pond", "water", "tank", "aquaculture", "limnology", "hatchery", "plankton", "bloom",
    "algae", "algal", "scum", "aerator", "aeration", "turbidity", "secchi", "sediment", 
    "sludge", "bottom", "benthic", "inlet", "outlet", "dyke", "acre", "acres", "depth",
    "harvest", "stocking", "culture", "biomass",
    "చెరువు", "నీరు", "ఏరేటర్", "ఏరేటర్లు", "లవణీయత", "మట్టి",
    # Telemetry Parameters
    "oxygen", "do", "ph", "alkalinity", "hardness", "ammonia", "tan", "nitrite", "nitrate",
    "salinity", "temperature", "temp", "orp", "transparency",
    "ఆక్సిజన్", "ఉష్ణోగ్రత", "ఉప్పుదనం", "క్షారత", "అమ్మోనియా", "ఆల్కలినిటీ",
    # Inputs & Treatments
    "feed", "feeding", "meal", "pellet", "bleaching", "permanganate", "kmno4", "chlorine", 
    "lime", "quicklime", "zeolite", "dolomite", "formalin", "probiotic", "chemical", "disinfectant",
    "copper", "sulfate", "paracetamol", "antibiotic", "biocide", "malachite", "mineral",
    "మేత", "సున్నం", "మందు", "జియోలైట్", "డోలమైట్",
    # Biosecurity & Diseases
    "pcr", "wssv", "ehp", "seed", "larvae", "post-larvae", "వైట్ స్పాట్"
}

# Explicit clinical signs
AQUACULTURE_CLINICAL_LEMMAS = {
    "disease", "mortality", "die", "dead", "dying", "gasp", "gasping", "suffocate",
    "sluggish", "erratic", "lesion", "ulcer", "rot", "spot", "whiteness", "necrosis",
    "anorexia", "off-feed", "lethargic", "foul", "odor", "smell", "fungus", "fungal", "infection",
    "చనిపోతున్నాయి", "మరణాలు", "కుళ్ళు", "తినడం లేదు", "ఈదలేకపోవడం", "లక్షణాలు", "మచ్చలు",
    # Romanized Telugu clinical signs
    "chastunnayi", "chaniptunnayi", "tinadam", "ledu", "baagoledu", "bagaledu", "baagolevu"
}

# Banned veterinary substances & toxic biocides (No \b on Telugu script!)
BANNED_SUBSTANCES_PATTERNS = [
    r"\bchloramphenicol\b",
    r"\bnitrofuran\b",
    r"\bnitrofurans\b",
    r"\bfurazolidone\b",
    r"\bmalachite green\b",
    r"\bendosulfan\b",
    r"\bpesticide\b",
    r"పురుగుమందు",
    r"క్లోరాంఫెనికాల్"
]

# Antibiotic prescription requests (Must be refused - System cannot prescribe drugs)
ANTIBIOTIC_PRESCRIPTION_PATTERNS = [
    r"\bantibiotic\b",
    r"\boxytetracycline\b",
    r"\bantifungal\b",
    r"యాంటీబయాటిక్",
    r"మందు పేరు"
]

# Therapeutic chemicals requiring expert veterinary prescription
THERAPEUTIC_CHEMICAL_PATTERNS = [
    r"\bformalin\b",
    r"\bcopper sulfate\b",
    r"\bcopper sulphate\b",
    r"\bcoppersulfate\b",
    r"\bpotassium permanganate\b",
    r"\bpermanganate\b",
    r"\bkmno4\b",
    r"ఫార్మలిన్",
    r"పొటాషియం పర్మాంగనేట్"
]

# Species / culture systems outside the brackishwater aquaculture corpus
OUT_OF_SCOPE_SYSTEMS_PATTERNS = [
    r"\bgoldfish\b",
    r"\bornamental\b",
    r"\bscampi\b",
    r"\bcage culture\b",
    r"\bin cages\b",
    r"\brohu and catla in freshwater\b",
    r"\brohu, catla\b",
    r"రోహు, కట్ల",
    r"రోహు",
    r"కట్ల"
]

# Unethical testing evasion & yield guarantees
UNETHICAL_OR_GUARANTEE_PATTERNS = [
    r"\bresidue test\b",
    r"\bpass the (?:antibiotic )?residue\b",
    r"\bhide (?:a )?(?:white spot|infection|disease)\b",
    r"\bbuyer\b.*?\bhide\b",
    r"తెలియకుండా ఎలా అమ్మాలి",
    r"తెలియకుండా",
    r"\bguarantee\b",
    r"\bguarantee ga\b",
    r"\bwill my pond get\b",
    r"\bvyaadhi vastunda\b",
    r"వ్యాధి వస్తుందా"
]

# Unproven folk remedies / unscientific claims
UNPROVEN_REMEDY_PATTERNS = [
    r"\bcommon salt\b.*?\bcure\b",
    r"\bneem leaves\b",
    r"వేప ఆకులు",
    r"\bpasupu\b",
    r"\bturmeric\b",
    r"\bcure all (?:shrimp|fish) diseases\b",
    r"\bcures? ehp in (?:three|\d+) days\b",
    r"\bdealer cheppadu\b"
]

# Multi-word chemical compounds
CHEMICAL_PATTERNS = {
    "bleach": [r"\bbleach\b", r"\bbleaching powder\b", r"\bhypochlorite\b", r"\bchlorine\b"],
    "permanganate": [r"\bpotassium permanganate\b", r"\bpermanganate\b", r"\bkmno4\b", r"పొటాషియం పర్మాంగనేట్"],
    "formalin": [r"\bformalin\b", r"\bformaldehyde\b", r"ఫార్మలిన్"],
    "copper_sulfate": [r"\bcopper sulfate\b", r"\bcopper sulphate\b", r"\bcoppersulfate\b", r"\bblue vitriol\b"],
    "ammonia": [r"\bammonia\b", r"\bammonium\b", r"అమ్మోనియా"],
    "probiotic": [r"\bprobiotic\b", r"\bprobiotics\b"]
}

# Dangerous combinations
HAZARDOUS_PAIRS = [
    ("bleach", "permanganate"),
    ("bleach", "formalin"),
    ("bleach", "ammonia"),
    ("formalin", "permanganate"),
    ("copper_sulfate", "formalin"),
    ("bleach", "probiotic")
]


def is_aquaculture_domain(doc: spacy.tokens.Doc, session_state: Optional[Dict[str, Any]] = None) -> bool:
    """Tier 1 Gate: Confirms inquiry belongs strictly to aquaculture."""
    text_lower = doc.text.lower()
    
    # 1. Terrestrial Agronomy Exclusions
    if any(m in text_lower for m in ["వరి", "వరి పంట", "పంటకు ఏ ఎరువు", "వరి పంటకు"]):
        if not any(aq in text_lower for aq in ["చెరువు", "చేప", "రొయ్య"]):
            return False

    terrestrial_markers = ["wheat", "paddy", "rice", "cotton", "sugarcane", "maize", "crop", "tractor", "potash", "cricket"]
    if any(re.search(rf"\b{m}\b", text_lower) for m in terrestrial_markers):
        if not any(aq in text_lower for aq in ["pond", "fish", "shrimp", "aquaculture", "cheruvu", "royya", "chepa"]):
            return False

    # 2. Direct Telugu aquaculture script markers
    telugu_in_domain = [
        "చేప", "రొయ్య", "చెరువు", "ఆక్సిజన్", "మేత", "చనిపో", "నీరు", "మందు",
        "పీత", "సున్నం", "జియోలైట్", "డోలమైట్", "లవణీయత", "ఆల్కలినిటీ", "అమ్మోనియా",
        "ఏరేటర్లు", "ఏరేటర్", "వైట్ స్పాట్", "మొప్పలు"
    ]
    if any(m in text_lower for m in telugu_in_domain):
        return True

    # 3. State Invariant: Slot-filling naked numbers or dimensions
    if session_state:
        if session_state.get("pending_metrics") or session_state.get("species"):
            if any(t.like_num or t.pos_ == "NUM" for t in doc):
                return True
            if any(w in text_lower for w in ["acre", "acres", "meter", "deep", "ganta", "gantalaki", "am", "pm"]):
                return True

    # 4. Water-quality telemetry is inherently in-domain
    has_telemetry_reading = any(c.isdigit() for c in text_lower) and any(
        u in text_lower for u in ["mg/l", "ppm", "ppt", "ph", "probe", "meter", "titration", "reads", "recalibrated", "celsius"]
    )
    if has_telemetry_reading:
        return True

    lemmas = {token.lemma_.lower() for token in doc}
    words = {token.text.lower() for token in doc}
    all_tokens = lemmas | words

    if bool(all_tokens & AQUACULTURE_DOMAIN_LEMMAS) or bool(all_tokens & AQUACULTURE_CLINICAL_LEMMAS):
        return True

    return False


def detect_chemical_hazards(doc: spacy.tokens.Doc) -> Optional[str]:
    """Tier 2 Gate: Detects dangerous chemical combinations, banned substances, and dosing."""
    text_lower = doc.text.lower()
    lemmas = {token.lemma_.lower() for token in doc}
    all_tokens = lemmas | {token.text.lower() for token in doc}

    # 1. Banned Substances & Toxic Biocide Intercept in Stocked Ponds
    if any(re.search(pat, text_lower) for pat in BANNED_SUBSTANCES_PATTERNS):
        return "BANNED_SUBSTANCE_REFUSAL"

    # 2. Antibiotic & Veterinary Prescription Refusal (Except when inquiry is fundamentally vague symptoms)
    has_vague_symptom = any(vd in text_lower for vd in ["don't look good", "not good", "సరిగా లేవు", "బాగా లేవు"])
    if any(re.search(pat, text_lower) for pat in ANTIBIOTIC_PRESCRIPTION_PATTERNS) and not has_vague_symptom:
        return "ANTIBIOTIC_PRESCRIPTION_REFUSAL"

    # 3. Species / Culture Systems Outside Corpus Scope (Goldfish, Scampi, Tilapia in Cages)
    if any(re.search(pat, text_lower) for pat in OUT_OF_SCOPE_SYSTEMS_PATTERNS):
        return "OUT_OF_SCOPE_REFUSAL"

    # 4. Unethical Lab Evasion & Unsupported Guarantees
    if any(re.search(pat, text_lower) for pat in UNETHICAL_OR_GUARANTEE_PATTERNS):
        return "UNSUPPORTED_GUARANTEE_REFUSAL"

    # 5. Unproven Folk Remedies & False Product Claims
    if any(re.search(pat, text_lower) for pat in UNPROVEN_REMEDY_PATTERNS):
        return "UNPROVEN_REMEDY_REFUSAL"

    # 6. Therapeutic Biocide Dosing Refusal (Formalin, CuSO4, KMnO4 for disease/algae -> ABSTAIN)
    dose_triggers = [
        "how many kg", "how much", "what dose", "dosage", "how much powder", "how many grams",
        "how many liters", "మోతాదు", "entha veyali", "ఎంత వాడాలి", "entha kalapali",
        "ఎన్ని కిలోలు వేయాలి", "ఎంత వేయాలి", "ఎంత సున్నం వేయాలి"
    ]
    has_dose_inquiry = any(dt in text_lower for dt in dose_triggers)

    has_disease_context = any(w in text_lower for w in ["disease", "fungus", "gill", "algae", "algal", "bloom", "మొప్పల", "వ్యాధికి"])
    if has_dose_inquiry and any(re.search(pat, text_lower) for pat in THERAPEUTIC_CHEMICAL_PATTERNS) and has_disease_context:
        return "THERAPEUTIC_DOSE_REFUSAL"

    # 7. Detect chemical families present via regex
    detected_chemicals = set()
    for chem_key, patterns in CHEMICAL_PATTERNS.items():
        if any(re.search(pat, text_lower) for pat in patterns):
            detected_chemicals.add(chem_key)

    # 8. Routine Soil/Water Conditioner Blind Dosing (Lime, Zeolite, Dolomite, Minerals -> CLARIFY)
    has_conditioner = bool(detected_chemicals) or any(c in all_tokens for c in {
        "chemical", "powder", "zeolite", "dolomite", "lime", "quicklime", "mandhu", "sunnam",
        "mineral", "minerals", "mineral mix", "జియోలైట్", "డోలమైట్", "సున్నం"
    }) or any(w in text_lower for w in ["mineral mix", "zeolite", "dolomite", "lime", "సున్నం", "జియోలైట్", "డోలమైట్", "potassium permanganate", "kmno4"])
    
    has_mortality = any(m in all_tokens for m in {"dying", "dead", "mortality", "losses", "die", "kill", "చనిపోతున్నాయి"})
    has_dimensions = any(d in text_lower for d in ["acre", "acres", "hectare", "meter deep", "depth", "volume", "m3", "tonnes"])

    if has_dose_inquiry and has_conditioner and (has_mortality or not has_dimensions):
        return "BLIND_DOSING_REFUSAL"

    # 9. Bleaching Powder in Stocked Ponds
    has_bleach = any(re.search(pat, text_lower) for pat in CHEMICAL_PATTERNS["bleach"])
    has_stocked = any(w in all_tokens for w in {"stocked", "stock", "shrimp", "fish", "culture", "standing", "రొయ్యలు", "చేపలు", "royyalu"})
    has_disinfect_stocked = (has_bleach and has_stocked) or any(w in text_lower for w in ["royyalu unna pond lo bleaching powder", "రొయ్యలు ఉన్న చెరువులో పురుగుమందు"])
    if has_disinfect_stocked and not any(w in all_tokens for w in {"preparation", "dry", "fallow", "pre-stocking", "before stocking"}):
        return "CHEMICAL_HAZARD"

    # 10. Hazardous Mixing Pairs
    for chem_a, chem_b in HAZARDOUS_PAIRS:
        if chem_a in detected_chemicals and chem_b in detected_chemicals:
            mix_markers = ["mix", "combine", "blend", "add", "together", "with", "and", "along with", "same day", "కలపడం"]
            if any(m in text_lower for m in mix_markers):
                return "CHEMICAL_HAZARD"

    return None


def detect_temporal_or_market_query(doc: spacy.tokens.Doc) -> bool:
    """Tier 3 Gate: Intercepts temporal commodity/market requests and administrative loans/subsidies."""
    text_lower = doc.text.lower()
    market_keywords = [
        "market price", "price today", "rate per kg", "cost per kg",
        "rate today", "bhimavaram today", "mandi price", "ధర",
        "best feed company", "best feed brand", "which feed brand",
        "which hatchery", "hatchery sells the best", "sells the best seed",
        "prices go up", "ధరలు పెరుగుతాయా", "rate entha", "today's price",
        # Financial & administrative out-of-scope queries
        "bank loan", "loan for", "subsidy", "crop insurance", "లోన్", "సబ్సిడీ"
    ]
    return any(kw in text_lower for kw in market_keywords)


def is_vague_complaint(doc: spacy.tokens.Doc) -> bool:
    """Tier 4 Gate: Intercepts ambiguous complaints lacking clinical symptoms or water telemetry."""
    text_lower = doc.text.lower().strip()
    has_telemetry = any(t.like_num or t.pos_ == "NUM" for t in doc)

    # 1. Unmeasured action/risk queries without numerical readings
    unmeasured_triggers = [
        "should i do a water exchange", "water exchange now", "water exchange cheyala",
        "can i increase the feed", "feed penchavacha",
        "is my ammonia level dangerous", "is my ph dangerous", "is my ph okay", "is my do okay",
        "ph సరిగా ఉందా", "do takkuva undi", "do is low", "surfacing in the morning",
        "పైకి వస్తున్నాయి", "రొయ్యలు పైకి", "tan is 2 mg/l. is it toxic?", "అమ్మోనియా 2 ఉంది"
    ]
    if any(trig in text_lower for trig in unmeasured_triggers) and not has_telemetry:
        return True
    if any(trig in text_lower for trig in ["tan is 2 mg/l. is it toxic", "అమ్మోనియా 2 ఉంది"]):
        # TAN toxicity strictly depends on missing pH and temperature
        if not ("ph" in text_lower and any(c.isdigit() for c in text_lower.split("ph")[-1][:8])):
            return True

    # 2. Ambiguous naked metrics lacking parameter identity
    if any(q in text_lower for q in [
        "the reading is", "రీడింగ్", "test lo", "meter lo", "my meter shows", "the value is"
    ]) and not any(p in text_lower for p in ["do", "ph", "ammonia", "tan", "nitrite", "salinity", "temperature", "alkalinity"]):
        return True

    # 3. Isolated symptoms without telemetry or context
    isolated_symptoms = [
        "turned reddish", "red colour", "soft shells", "shell soft",
        "white strings", "black gills", "మొప్పలు నల్లగా", "red patches",
        "white spots on the shell", "తెల్ల మచ్చలు"
    ]
    if any(s in text_lower for s in isolated_symptoms) and not has_telemetry:
        return True

    # 4. Emotive distress / general vague descriptors
    vague_descriptors = [
        "bad", "strange", "weird", "dirty", "unusual", "not good", "problem",
        "baagoledu", "bagaledu", "baagolevu", "not doing well", "doing well",
        "edo problem", "something seems off", "something is wrong", "సరిగా లేవు", "సమస్య ఉంది",
        "growth raavadam ledu", "growth is poor", "పెరగడం లేదు", "correct ga levu", "don't look good",
        "ఏదో సమస్య ఉంది", "బాగా లేవు", "చేపలు బాగోలేవు", "రొయ్యలు సరిగా లేవు"
    ]
    has_vague_descriptor = any(vd in text_lower for vd in vague_descriptors)
    
    has_clinical_symptoms = any(
        s in text_lower for s in [
            "gasping", "mortality", "dead", "dying", "lesion",
            "rot", "smell", "odor", "floating", "sluggish", "erratic", "green", "turbid", "surface"
        ]
    )

    has_entity = any(e in text_lower for e in ["pond", "water", "shrimp", "fish", "stock", "royyalu", "chepalu", "cheruvu", "harvest"])
    if (has_entity or "royyalu" in text_lower or "chepalu" in text_lower or any(c in text_lower for c in ["చేపలు", "రొయ్యలు"])) and has_vague_descriptor:
        if not has_telemetry and not has_clinical_symptoms:
            return True

    return False


def route_query_policy(raw_query: str, session_state: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
    """Primary routing decision engine."""
    q_clean = raw_query.strip()
    doc = nlp(q_clean)

    # Tier 1: Aquaculture Domain Gate (State Aware)
    if not is_aquaculture_domain(doc, session_state=session_state):
        return "POLICY_INTERCEPT", "OUT_OF_DOMAIN"

    # Tier 2: Chemical Safety, Therapeutic Dosing, & Banned Substances
    hazard_key = detect_chemical_hazards(doc)
    if hazard_key:
        return "POLICY_INTERCEPT", hazard_key

    # Tier 3: Temporal / Real-time Commodity Limitation
    if detect_temporal_or_market_query(doc):
        return "POLICY_INTERCEPT", "REALTIME_DATA_LIMITATION"

    # Tier 4: Vague Diagnostic & Missing Context Gate
    if is_vague_complaint(doc):
        return "POLICY_INTERCEPT", "VAGUE_DIAGNOSTIC"

    return "PASS", None


def evaluate_policy_tiers(query: str, session_state: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """Compatibility wrapper adapting route_query_policy to benchmark suite."""
    route_status, key = route_query_policy(query, session_state=session_state)
    if route_status == "POLICY_INTERCEPT":
        return True, key, None
    return False, None, None