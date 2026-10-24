import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.diagnostic_engine import compute_unionized_ammonia, evaluate_telemetry_health_and_conflicts

def test_rc02():
    print("=" * 60)
    print("TESTING RC-02: EMERSON AMMONIA & FAST-PATH ALERT")
    print("=" * 60)

    # 1. Test Emerson Math: TAN 3.5 mg/L, pH 8.8 at 28C
    nh3 = compute_unionized_ammonia(3.5, 8.8, 28.0)
    print(f"Calculated toxic NH3 for TAN=3.5, pH=8.8: {nh3:.4f} mg/L")
    assert nh3 >= 0.05, f"Expected toxic NH3 >= 0.05, got {nh3}"
    print("[PASS] Emerson equation correctly identifies toxic threshold.")

    # 2. Test Fast-Path Telemetry Reflex
    state = {
        "species": "shrimp",
        "fresh_telemetry_in_turn": True,
        "telemetry": {
            "TAN": [{"value": 3.5, "unit": "mg/L"}],
            "pH": [{"value": 8.8, "unit": ""}]
        }
    }
    has_conflict, intercept_type, msg = evaluate_telemetry_health_and_conflicts(state)
    print(f"Fast-Path Intercept: {intercept_type} | {msg}")
    assert has_conflict is True
    assert intercept_type == "EMERGENCY_AMMONIA"
    print("[PASS] Emergency ammonia reflex fires deterministically.")

    # 3. Test Non-Toxic Level (TAN 0.5, pH 7.2)
    safe_state = {
        "species": "shrimp",
        "fresh_telemetry_in_turn": True,
        "telemetry": {
            "TAN": [{"value": 0.5, "unit": "mg/L"}],
            "pH": [{"value": 7.2, "unit": ""}]
        }
    }
    has_conflict, intercept_type, _ = evaluate_telemetry_health_and_conflicts(safe_state)
    assert has_conflict is False
    print("[PASS] Safe ammonia levels do not trigger false emergency sirens.")

    print("=" * 60)
    print("ALL RC-02 TESTS PASSED CLEANLY.")

if __name__ == "__main__":
    test_rc02()