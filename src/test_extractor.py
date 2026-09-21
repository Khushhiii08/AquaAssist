from intent_extractor import extract_hypothesis

# The exact failing cases from the Phase 3 Testing Report
TEST_CASES = [
    {
        "id": "T2", 
        "input": "My shrimp are swimming near the surface and gasping for air.",
        "must_not_contain": ["not"] # Catches the meaning reversal observed in T2
    },
    {
        "id": "T7", 
        "input": "My shrimp are gasping at the surface, and the dissolved oxygen is 2 mg/L.",
        "must_not_contain": ["should", "check", "recommend"] # Catches the added advice observed in T7
    },
    {
        "id": "T11",
        "input": "My shrimp are dying. Tell me exactly which chemical and dosage I should add immediately.",
        "must_not_contain": ["due to", "lack of", "feeding"] # Catches the invented causes observed in T11
    },
    {
        "id": "T10",
        "input": "I have not measured the dissolved oxygen.",
        "must_not_contain": ["feeding", "color", "colour"] # Catches the unsupported details added in T10
    }
]

def run_evals():
    print("[*] Running Extractor Evals...\n")
    for test in TEST_CASES:
        print(f"--- Test {test['id']} ---")
        print(f"Input: {test['input']}")
        
        result = extract_hypothesis(test['input'])
        print(f"Output: {result}")
        
        # The automated validation step
        failed = any(bad_word in result.lower() for bad_word in test['must_not_contain'])
        if failed:
            print("❌ FAIL: Hallucination or meaning reversal detected.\n")
        else:
            print("✅ PASS: Clean extraction.\n")

if __name__ == "__main__":
    run_evals()