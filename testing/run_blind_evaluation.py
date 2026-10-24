"""
AquaAssist: Blind Test Batch Evaluation Runner (AQ-001 to AQ-258)
Executes full regression evaluation across the 4 member query workbooks.
Calibrated scoring harness resolving RC-06:
- SENSOR_CONFLICT -> ABSTAIN (Physically implausible biological bounds)
- CONFLICT_READINGS -> CLARIFY (Concurrent diverging sensors)
- REALTIME_DATA_LIMITATION -> ABSTAIN (Market/pricing boundaries)
- BLIND_DOSING_REFUSAL -> CLARIFY (Missing pond volume/depth)
- VAGUE_DIAGNOSTIC -> CLARIFY (Missing clinical observables)
- EMERGENCY_HYPOXIA / EMERGENCY_AMMONIA -> ANSWER (Immediate emergency protocol)
"""

import os
import sys
import glob
import time
import json
import pandas as pd
from typing import Dict, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.intent_extractor import extract_hypothesis
from src.decision_router import route_query_policy
from src.diagnostic_engine import (
    load_engine,
    run_diagnostic,
    evaluate_telemetry_health_and_conflicts
)

# Deterministic fast-path intercept to canonical 3-class decision mapping
INTERCEPT_TO_DECISION = {
    # ABSTAIN (Safety boundaries, biological impossibilities, prohibited practices, out of scope)
    "OUT_OF_DOMAIN": "ABSTAIN",
    "REALTIME_DATA_LIMITATION": "ABSTAIN",
    "CHEMICAL_HAZARD": "ABSTAIN",
    "BANNED_SUBSTANCE_REFUSAL": "ABSTAIN",
    "ANTIBIOTIC_PRESCRIPTION_REFUSAL": "ABSTAIN",
    "THERAPEUTIC_DOSE_REFUSAL": "ABSTAIN",
    "OUT_OF_SCOPE_REFUSAL": "ABSTAIN",
    "UNSUPPORTED_GUARANTEE_REFUSAL": "ABSTAIN",
    "UNPROVEN_REMEDY_REFUSAL": "ABSTAIN",
    "SENSOR_CONFLICT": "ABSTAIN",

    # CLARIFY (Missing info, unmeasured complaints, marginal bands, or diverging sensors)
    "CONFLICT_READINGS": "CLARIFY",
    "BLIND_DOSING_REFUSAL": "CLARIFY",
    "VAGUE_DIAGNOSTIC": "CLARIFY",
    "MARGINAL_DO_CLARIFY": "CLARIFY",

    # ANSWER (Acute actionable emergency interventions)
    "EMERGENCY_HYPOXIA": "ANSWER",
    "EMERGENCY_AMMONIA": "ANSWER",
}


def load_all_queries() -> pd.DataFrame:
    """Discovers and concatenates Member 1 through Member 4 Excel workbooks."""
    search_paths = [
        os.path.join(PROJECT_ROOT, "messy_query_excelsheets", "AquaAssist_Queries_Member_*.xlsx"),
        os.path.join(PROJECT_ROOT, "AquaAssist_Queries_Member_*.xlsx"),
        os.path.join(PROJECT_ROOT, "data", "AquaAssist_Queries_Member_*.xlsx"),
        os.path.join(CURRENT_DIR, "AquaAssist_Queries_Member_*.xlsx"),
    ]
    files = []
    for pattern in search_paths:
        matched = glob.glob(pattern)
        if matched:
            files = sorted(matched)
            break

    if not files:
        raise FileNotFoundError(
            "Could not locate AquaAssist_Queries_Member_1..4.xlsx. "
            "Please ensure they are placed in messy_query_excelsheets/ or project root."
        )

    dfs = []
    for f in files:
        df = pd.read_excel(f, sheet_name="Queries")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[*] Loaded {len(combined)} queries across {len(files)} member sheets.")
    return combined


def evaluate_single_query(
    row: pd.Series,
    embedding_model,
    nli_model,
    collection
) -> Dict[str, Any]:
    q_id = row["ID"]
    query = str(row["Farmer query"]).strip()
    expected_decision = str(row["Draft decision (Claude)"]).strip().upper()
    expected_path = str(row["Expected path"]).strip().upper()
    category = str(row["Category"]).strip()
    
    # Initialize state, handling multi-turn context if present
    session_state = {}
    prev_context = str(row.get("Previous turn (multi-turn context)", "")).strip()
    if prev_context and prev_context != "nan":
        if "seabass" in prev_context.lower():
            session_state["species"] = "fish"
        elif "shrimp" in prev_context.lower() or "royyalu" in prev_context.lower():
            session_state["species"] = "shrimp"
        elif "fish" in prev_context.lower() or "chepalu" in prev_context.lower():
            session_state["species"] = "fish"
            
        # Set pending slot expecting metrics
        if "area and water depth" in prev_context.lower():
            session_state["pending_metrics"] = ["dimensions"]
        elif "do reading" in prev_context.lower() or "do value" in prev_context.lower():
            session_state["pending_metrics"] = ["DO"]
        elif "ph and temperature" in prev_context.lower():
            session_state["pending_metrics"] = ["pH", "temperature"]

    # Layer 1: Ingestion
    hypothesis, parsed_state = extract_hypothesis(query, session_state=session_state)
    parsed_state["raw_query"] = query

    # Layer 2: Fast-Path Policy & Conflict Gates
    pol_status, pol_key = route_query_policy(query, session_state=parsed_state)
    has_conflict, conflict_type, conflict_msg = evaluate_telemetry_health_and_conflicts(parsed_state)

    is_fast_path = (pol_status == "POLICY_INTERCEPT") or has_conflict
    system_path = "FAST" if is_fast_path else "SLOW"

    system_decision = "UNKNOWN"
    intercept_key = None

    if is_fast_path:
        intercept_key = pol_key or conflict_type
        system_decision = INTERCEPT_TO_DECISION.get(intercept_key, "ABSTAIN")
    else:
        # Layer 3: Slow-Path Diagnostic RAG
        diag_res = run_diagnostic(
            query,
            parsed_state,
            embedding_model,
            nli_model,
            collection
        )
        system_decision = diag_res["decision"]

    decision_correct = (system_decision == expected_decision)
    path_correct = (system_path == expected_path)

    return {
        "id": q_id,
        "query": query,
        "category": category,
        "expected_decision": expected_decision,
        "system_decision": system_decision,
        "expected_path": expected_path,
        "system_path": system_path,
        "intercept_key": intercept_key,
        "decision_correct": decision_correct,
        "path_correct": path_correct
    }


def run_batch_evaluation():
    print("=" * 80)
    print("  AquaAssist: Calibrated Blind Batch Evaluation Runner (AQ-001 - AQ-258)")
    print("=" * 80)

    df_queries = load_all_queries()

    print("[*] Booting diagnostic routing models and connecting to ChromaDB...")
    embedding_model, nli_model, collection = load_engine()

    results = []
    correct_decisions = 0
    correct_paths = 0
    total = len(df_queries)

    print(f"\n[*] Evaluating {total} benchmark queries...\n")
    t0 = time.perf_counter()

    for idx, (_, row) in enumerate(df_queries.iterrows(), start=1):
        res = evaluate_single_query(row, embedding_model, nli_model, collection)
        results.append(res)

        if res["decision_correct"]:
            correct_decisions += 1
        if res["path_correct"]:
            correct_paths += 1

        mark = "[PASS]" if res["decision_correct"] else "[FAIL]"
        if idx % 25 == 0 or idx == total or not res["decision_correct"]:
            print(
                f"{mark} [{res['id']}] Exp: {res['expected_decision']:<7} | "
                f"Got: {res['system_decision']:<7} | Path: {res['system_path']:<4} | "
                f"Cat: {res['category'][:32]}"
            )

    elapsed = time.perf_counter() - t0
    dec_acc = (correct_decisions / total) * 100.0
    path_acc = (correct_paths / total) * 100.0

    print("\n" + "=" * 80)
    print("  BLIND EVALUATION SCORECARD REPORT")
    print("=" * 80)
    print(f" Total Evaluated Queries: {total}")
    print(f" Execution Time:          {elapsed:.2f} s ({elapsed / total * 1000:.1f} ms/query)")
    print(f" Decision Accuracy:       {correct_decisions}/{total} ({dec_acc:.2f}%)")
    print(f" Path Routing Accuracy:   {correct_paths}/{total} ({path_acc:.2f}%)")
    print("=" * 80)

    # Breakdown by Category
    res_df = pd.DataFrame(results)
    cat_summary = res_df.groupby("category").agg(
        total=("id", "count"),
        correct=("decision_correct", "sum")
    )
    cat_summary["accuracy_%"] = (cat_summary["correct"] / cat_summary["total"]) * 100.0
    print("\nAccuracy Breakdown by Query Category:")
    print(cat_summary.to_string())

    out_json = os.path.join(PROJECT_ROOT, "blind_eval_results_258.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_queries": total,
                "correct_decisions": correct_decisions,
                "decision_accuracy_percentage": dec_acc,
                "path_routing_accuracy_percentage": path_acc,
                "elapsed_seconds": elapsed,
                "results": results
            },
            f,
            indent=2,
            ensure_ascii=False
        )
    print(f"\n[+] Detailed evaluation log exported to: {out_json}")


if __name__ == "__main__":
    run_batch_evaluation()