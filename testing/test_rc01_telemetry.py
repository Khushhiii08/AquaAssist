"""
AquaAssist Verification Suite: RC-01 Telemetry Extraction Invariants
Tests that syntactic parameters are never overwritten by mg/L or ppm units,
that hardware identifiers ('meter', 'probe') are decoupled from DO, and that
timestamps ('4 am', '6 pm') are never parsed as sensor telemetry.
"""

import os
import sys

# Resolve project root from 'testing/'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.intent_extractor import nlp, extract_telemetry
def run_rc01_regression_suite():
    test_cases = [
        # --- Group 1: Non-DO Parameters with mg/L or ppm (Was triggering False Hypoxia) ---
        {
            "id": "AQ-017",
            "query": "Nitrite is 1.5 mg/L in my shrimp pond",
            "expected_param": "NO2",
            "expected_val": 1.5,
            "forbidden_param": "DO",
            "desc": "Nitrite with mg/L must bind as NO2, never DO"
        },
        {
            "id": "AQ-043",
            "query": "TAN level is 2.8 mg/l and water is cloudy",
            "expected_param": "TAN",
            "expected_val": 2.8,
            "forbidden_param": "DO",
            "desc": "TAN with mg/l must bind as TAN, never DO"
        },
        {
            "id": "AQ-079",
            "query": "Water test shows nitrite 8.8 ppm",
            "expected_param": "NO2",
            "expected_val": 8.8,
            "forbidden_param": "DO",
            "desc": "Nitrite with ppm must bind as NO2, never DO"
        },
        {
            "id": "AQ-197",
            "query": "Nitrate is 25 mg/L in our nursery pond",
            "expected_param": "NO3",
            "expected_val": 25.0,
            "forbidden_param": "DO",
            "desc": "Nitrate with mg/L must bind as NO3, never DO"
        },

        # --- Group 2: Alkalinity Values (Was triggering False Impossible DO > 22 mg/L) ---
        {
            "id": "AQ-173",
            "query": "Alkalinity is 110 mg/L in vannamei culture",
            "expected_param": "alkalinity",
            "expected_val": 110.0,
            "forbidden_param": "DO",
            "desc": "Alkalinity 110 mg/L must not bind as DO (which would reject DO > 22)"
        },
        {
            "id": "AQ-208",
            "query": "Pond alkalinity is 130 ppm",
            "expected_param": "alkalinity",
            "expected_val": 130.0,
            "forbidden_param": "DO",
            "desc": "Alkalinity 130 ppm must retain alkalinity key"
        },

        # --- Group 3: Hardware Words 'Probe' & 'Meter' (Was mapping to DO >= 5.0) ---
        {
            "id": "AQ-105",
            "query": "pH probe says 6.2 and test kit says 8.0",
            "expected_param": "pH",
            "forbidden_param": "DO",
            "desc": "pH probe must bind to pH, probe must never alias to DO"
        },
        {
            "id": "AQ-118",
            "query": "pH kit 8.1 meter 6.4",
            "expected_param": "pH",
            "forbidden_param": "DO",
            "desc": "pH meter must bind to pH, meter must never alias to DO"
        },

        # --- Group 4: Timestamps with DO (Was overwriting DO values) ---
        {
            "id": "AQ-111",
            "query": "DO is 2.1 mg/L at 4 am",
            "expected_param": "DO",
            "expected_val": 2.1,
            "forbidden_val": 4.0,
            "desc": "4 am must be ignored; DO value must strictly be 2.1"
        },
        {
            "id": "AQ-148",
            "query": "DO is 6.2 mg/L at 6 am",
            "expected_param": "DO",
            "expected_val": 6.2,
            "forbidden_val": 6.0,
            "desc": "6 am must be ignored; DO value must strictly be 6.2"
        },

        # --- Group 5: Native Script & Telugu Timestamp ---
        {
            "id": "AQ-089-Stem",
            "query": "DO 3.5 mg/L at 5 gantalaki",
            "expected_param": "DO",
            "expected_val": 3.5,
            "forbidden_val": 5.0,
            "desc": "Telugu hour marker 'gantalaki' must not be captured as telemetry value 5.0"
        }
    ]

    passed = 0
    failed = 0

    print("=" * 75)
    print("RUNNING AQUAASSIST RC-01 TELEMETRY INVARIANT AUDIT SUITE")
    print("=" * 75)

    for case in test_cases:
        doc = nlp(case["query"])
        records = extract_telemetry(doc)

        # Check for presence of extracted parameters
        extracted_params = [r["parameter"] for r in records]
        extracted_vals = [r["value"] for r in records]

        case_passed = True
        failure_reasons = []

        # Invariant 1: Expected parameter must exist
        if case["expected_param"] not in extracted_params:
            case_passed = False
            failure_reasons.append(f"Missing expected parameter '{case['expected_param']}'. Found: {extracted_params}")

        # Invariant 2: Forbidden parameter (e.g. DO) must NOT exist
        if "forbidden_param" in case and case["forbidden_param"] in extracted_params:
            # Only fail if it wrongly replaced the expected one or was falsely introduced
            if case["expected_param"] != case["forbidden_param"]:
                case_passed = False
                failure_reasons.append(f"Forbidden parameter '{case['forbidden_param']}' was falsely extracted!")

        # Invariant 3: Expected value must match
        if "expected_val" in case:
            matching_vals = [r["value"] for r in records if r["parameter"] == case["expected_param"]]
            if not matching_vals or case["expected_val"] not in matching_vals:
                case_passed = False
                failure_reasons.append(f"Value mismatch for '{case['expected_param']}': expected {case['expected_val']}, got {matching_vals}")

        # Invariant 4: Forbidden value (e.g. timestamp numbers like 4.0 or 6.0) must NOT exist
        if "forbidden_val" in case:
            if case["forbidden_val"] in extracted_vals:
                case_passed = False
                failure_reasons.append(f"Timestamp number {case['forbidden_val']} was falsely captured as telemetry value!")

        if case_passed:
            passed += 1
            print(f"[PASS] {case['id']}: {case['desc']}")
        else:
            failed += 1
            print(f"[FAIL] {case['id']}: {case['desc']}")
            for reason in failure_reasons:
                print(f"       -> {reason}")
            print(f"       Raw Extracted: {records}")

    print("-" * 75)
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED across {len(test_cases)} safety scenarios.")
    print("=" * 75)

    if failed > 0:
        sys.exit(1)
    else:
        print("ALL RC-01 INVARIANTS VERIFIED CLEAN. Safe to proceed to Task 2.")

if __name__ == "__main__":
    run_rc01_regression_suite()