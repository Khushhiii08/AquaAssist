import json
import random
import re
from pathlib import Path


# ============================================================
# AquaAssist - RLAIF Dataset Generator
# ============================================================

INPUT_FILE = Path("data/processed/chunks.json")
OUTPUT_DIR = Path("data/evaluation")
OUTPUT_FILE = OUTPUT_DIR / "rlaif_dataset.json"

random.seed(42)


# ------------------------------------------------------------
# Biological safety checks
# ------------------------------------------------------------

def verify_aquaculture_invariants(text):
    """
    Returns True when the text does not contain obviously
    impossible aquaculture water-quality values.
    """

    text_lower = text.lower()

    # Check pH values.
    ph_patterns = re.findall(
        r"\bph\s*(?:=|:|of|is)?\s*(\d+(?:\.\d+)?)",
        text_lower
    )

    for value in ph_patterns:
        ph = float(value)

        # Conservative biological sanity range.
        if ph < 6.0 or ph > 9.5:
            return False

    # Check dissolved oxygen (DO).
    do_patterns = re.findall(
        r"\b(?:do|dissolved oxygen)\s*(?:=|:|of|is)?\s*(-?\d+(?:\.\d+)?)",
        text_lower
    )

    for value in do_patterns:
        do_value = float(value)

        # Negative dissolved oxygen is physically impossible.
        if do_value < 0:
            return False

    return True


# ------------------------------------------------------------
# Telugu query generation
# ------------------------------------------------------------

def make_telugu_query(chunk):
    """
    Creates a simple Telugu query from the chunk metadata.
    The original evidence remains in English.
    """

    domain = chunk.get("domain_category", "aquaculture")
    species = chunk.get("target_species", "general aquaculture")

    templates = [
        f"{species} కోసం {domain} గురించి ఏమి తెలుసుకోవాలి?",
        f"{domain} విషయంలో {species} కోసం సరైన సమాచారం ఏమిటి?",
        f"{species} పెంపకంలో {domain} ఎందుకు ముఖ్యమైనది?",
    ]

    return random.choice(templates)


# ------------------------------------------------------------
# Triplet generation
# ------------------------------------------------------------

def create_triplets(chunks):
    dataset = []

    valid_chunks = []

    for index, chunk in enumerate(chunks):

        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        if not verify_aquaculture_invariants(text):
            continue

        valid_chunks.append((index, chunk))

    for position, (index, chunk) in enumerate(valid_chunks):

        text = str(chunk.get("text", "")).strip()

        source = chunk.get(
            "source_document",
            "unknown_document"
        )

        domain = chunk.get(
            "domain_category",
            "General Aquaculture"
        )

        species = chunk.get(
            "target_species",
            "General Aquaculture"
        )

        query = make_telugu_query(chunk)

        chunk_id = f"chunk_{index}"

        # -----------------------------
        # Entailment
        # -----------------------------
        entailment = {
            "query": query,
            "evidence": text,
            "label": "Entailment",
            "label_id": 1,
            "evidence_chunk_id": chunk_id,
            "source_document": source,
            "domain_category": domain,
            "target_species": species,
        }

        dataset.append(entailment)

        # -----------------------------
        # Neutral
        # -----------------------------
        neutral_query = (
            f"{species} కోసం {domain} మరియు "
            f"చేపల పెంపకం మధ్య సంబంధం ఏమిటి?"
        )

        neutral = {
            "query": neutral_query,
            "evidence": text,
            "label": "Neutral",
            "label_id": 0,
            "evidence_chunk_id": chunk_id,
            "source_document": source,
            "domain_category": domain,
            "target_species": species,
        }

        dataset.append(neutral)

        # -----------------------------
        # Contradiction
        # -----------------------------
        contradiction_query = (
            f"{species} కోసం {domain} విషయంలో "
            f"ఈ సమాచారానికి విరుద్ధమైన పరిస్థితి సరైనదా?"
        )

        contradiction = {
            "query": contradiction_query,
            "evidence": text,
            "label": "Contradiction",
            "label_id": -1,
            "evidence_chunk_id": chunk_id,
            "source_document": source,
            "domain_category": domain,
            "target_species": species,
        }

        dataset.append(contradiction)

    return dataset


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("[*] Loading processed chunks...")

    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        print("[ERROR] chunks.json must contain a JSON list.")
        return

    print(f"[*] Loaded {len(chunks)} chunks.")

    dataset = create_triplets(chunks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            dataset,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("[*] SUCCESS: RLAIF dataset generated!")
    print(f"[*] Valid triplets: {len(dataset)}")
    print(f"[*] Saved to: {OUTPUT_FILE}")

    # Dataset summary
    counts = {
        "Entailment": 0,
        "Neutral": 0,
        "Contradiction": 0
    }

    for item in dataset:
        label = item["label"]

        if label in counts:
            counts[label] += 1

    print()
    print("[*] Label distribution:")
    print(f"    Entailment:   {counts['Entailment']}")
    print(f"    Neutral:      {counts['Neutral']}")
    print(f"    Contradiction: {counts['Contradiction']}")


if __name__ == "__main__":
    main()
