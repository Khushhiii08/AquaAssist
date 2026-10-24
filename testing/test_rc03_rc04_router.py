"""
AquaAssist Verification Suite: Tasks 3 & 4 (RC-03 & RC-04)
Tests Romanized Telugu stems, state-aware slot replies, banned substance intercepts,
bleaching in stocked ponds, and expanded vague diagnosis gates.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.decision_router import route_query_policy

def run_tests():
    print("=" * 70)
    print("TESTING RC-03 & RC-04 DECISION ROUTER INVARIANTS")
    print("=" * 70)

    test_cases = [
        # --- Task 3 (RC-03): Romanized Telugu Stems (Must NOT be OUT_OF_DOMAIN) ---
        {
            "id": "AQ-009",
            "query": "Naa cheruvulo royyalu feed tinadam ledu",
            "state": None,
            "expected_status": "PASS",  # Has clinical symptom ('tinadam ledu'), in domain
            "desc": "Romanized Telugu (cheruvulo, royyalu) must pass Tier 1"
        },
        {
            "id": "AQ-010",
            "query": "Mud crab culture lo salinity entha undali?",
            "expected_status": "PASS",
            "state": None,
            "desc": "Mud crab and Romanized 'entha' must be recognized in-domain"
        },
        {
            "id": "AQ-038-Slot",
            "query": "6.1",
            "state": {"pending_metrics": ["pH"], "species": "fish"},
            "expected_status": "PASS",
            "desc": "Naked number slot reply with pending_metrics must be accepted"
        },
        {
            "id": "AQ-013-Slot",
            "query": "2 acres, about 1.2 m deep",
            "state": {"pending_metrics": ["dimensions"], "species": "shrimp"},
            "expected_status": "PASS",
            "desc": "Dimensional slot reply with active state must be accepted"
        },

        # --- Task 4 (RC-04): Banned Substances (Must trigger intercept) ---
        {
            "id": "AQ-008",
            "query": "Can I use malachite green to treat fungal infection?",
            "state": None,
            "expected_status": "POLICY_INTERCEPT",
            "expected_key": "BANNED_SUBSTANCE_REFUSAL",
            "desc": "Malachite green must trigger deterministic banned substance refusal"
        },
        {
            "id": "AQ-Banned-Chlor",
            "query": "How much chloramphenicol should I add for vibrio in shrimp?",
            "state": None,
            "expected_status": "POLICY_INTERCEPT",
            "expected_key": "BANNED_SUBSTANCE_REFUSAL",
            "desc": "Chloramphenicol must trigger deterministic banned substance refusal"
        },

        # --- Task 4 (RC-04): Dangerous Mixtures & Bleach in Stocked Ponds ---
        {
            "id": "AQ-018",
            "query": "Can I add bleaching powder to my stocked shrimp pond to disinfect water?",
            "state": None,
            "expected_status": "POLICY_INTERCEPT",
            "expected_key": "CHEMICAL_HAZARD",
            "desc": "Bleaching powder in stocked ponds must trigger chemical hazard"
        },
        {
            "id": "AQ-026",
            "query": "Can I apply bleaching powder and probiotics on the same day?",
            "state": None,
            "expected_status": "POLICY_INTERCEPT",
            "expected_key": "CHEMICAL_HAZARD",
            "desc": "Bleach + Probiotic same-day combination must trigger chemical hazard"
        },

        # --- Task 4 (RC-04): Expanded Vague Diagnostic Queries ---
        {
            "id": "AQ-Vague-Shrimp",
            "query": "My shrimp are not doing well",
            "state": None,
            "expected_status": "POLICY_INTERCEPT",
            "expected_key": "VAGUE_DIAGNOSTIC",
            "desc": "Stock-level vague complaint ('shrimp not doing well') must trigger VAGUE_DIAGNOSTIC"
        },
        {
            "id": "AQ-131-NoMetric",
            "query": "Is my ammonia level dangerous?",
            "state": None,
            "expected_status": "POLICY_INTERCEPT",
            "expected_key": "VAGUE_DIAGNOSTIC",
            "desc": "Unmeasured risk inquiry ('is my ammonia dangerous?') must trigger VAGUE_DIAGNOSTIC"
        }
    ]

    passed = 0
    failed = 0

    for case in test_cases:
        status, key = route_query_policy(case["query"], session_state=case.get("state"))
        
        ok = (status == case["expected_status"])
        if ok and "expected_key" in case:
            ok = (key == case["expected_key"])

        if ok:
            passed += 1
            print(f"[PASS] {case['id']}: {case['desc']}")
        else:
            failed += 1
            print(f"[FAIL] {case['id']}: {case['desc']}")
            print(f"       Expected: ({case['expected_status']}, {case.get('expected_key')}) | Got: ({status}, {key})")

    print("-" * 70)
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED out of {len(test_cases)} tests.")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print("ALL RC-03 & RC-04 ROUTING INVARIANTS VERIFIED CLEAN.")

if __name__ == "__main__":
    run_tests()