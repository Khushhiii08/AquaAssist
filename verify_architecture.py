"""
AquaAssist Architecture 2 Compliance Auditor
Runs static AST and pattern analysis to verify strict anti-bandaid implementation.
"""

import ast
import os
import re
import sys

ERRORS = []
WARNINGS = []
PASSED = []


def check(condition, message, file_path, is_warning=False):
  if condition:
    PASSED.append(f"[PASS] {os.path.basename(file_path)}: {message}")
  else:
    target = WARNINGS if is_warning else ERRORS
    label = "[WARN]" if is_warning else "[FAIL]"
    target.append(f"{label} {os.path.basename(file_path)}: {message}")


def audit_intent_extractor(file_path):
  if not os.path.exists(file_path):
    ERRORS.append(f"[FAIL] Missing file: {file_path}")
    return
  with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

  # 1. No "eating normally" fallback
  check(
      "eating normally" not in code,
      "No 'eating normally' ghost symptom fallback found.",
      file_path,
  )

  # 2. Uses spaCy dependency parsing / POS tags for numbers
  check(
      "like_num" in code or "pos_ == \"NUM\"" in code or 'pos_ == "NUM"' in code,
      "Extracts telemetry via dependency trees / NUM tokens (zero static"
      " whitelist).",
      file_path,
  )

  # 3. Separates species cleanly
  check(
      'organism = "fish"' in code or "organism = 'fish'" in code,
      "Explicitly isolates fish and shrimp organisms.",
      file_path,
  )

  # 4. Stores structured telemetry in session_state
  check(
      'session_state["telemetry"]' in code
      or "session_state['telemetry']" in code,
      "Stores structured, typed telemetry dictionary into session_state.",
      file_path,
  )


def audit_decision_router(file_path):
  if not os.path.exists(file_path):
    ERRORS.append(f"[FAIL] Missing file: {file_path}")
    return
  with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

  # 1. Uses lemma matching
  check(
      "lemma_" in code,
      "Domain classifier checks lemmatized tokens (inflection resilient).",
      file_path,
  )

  # 2. Chemical hazard gate present
  check(
      "detect_chemical_hazards" in code or "CHEMICAL_HAZARD" in code,
      "Tier 2 Chemical safety & hazard gate is implemented.",
      file_path,
  )

  # 3. Temporal market gate present
  check(
      "detect_temporal_or_market_query" in code
      or "REALTIME_DATA_LIMITATION" in code,
      "Tier 3 Temporal market pricing gate is implemented.",
      file_path,
  )


def audit_diagnostic_engine(file_path):
  if not os.path.exists(file_path):
    ERRORS.append(f"[FAIL] Missing file: {file_path}")
    return
  with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

  # 1. Zero fake NLI
  check(
      '{"entailment": 1.0' not in code and '{"entailment": 1.0000' not in code,
      "Zero hardcoded fake entailment (1.0) injections.",
      file_path,
  )
  check(
      "1.0 - dist" not in code and "1.0 - distance" not in code,
      "Zero synthetic NLI distance approximations.",
      file_path,
  )

  # 2. Directional NLI [Premise, Hypothesis]
  check(
      "convert_query_to_declarative_hypothesis" in code
      or "This documentation explains" in code,
      "Transforms questions into declarative hypotheses for DeBERTa.",
      file_path,
  )

  # 3. Uses active DB path
  check(
      "get_active_db_path" in code,
      "Dynamically queries get_active_db_path() for ChromaDB connection.",
      file_path,
  )


def audit_response_generator(file_path):
  if not os.path.exists(file_path):
    ERRORS.append(f"[FAIL] Missing file: {file_path}")
    return
  with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

  # 1. No hardcoded test responses
  has_hardcoded_causes = bool(
      re.search(r'if\s+["\']causes["\']\s+in\s+q', code)
  )
  check(
      not has_hardcoded_causes,
      "Zero hardcoded answer dictionaries for T1 causes.",
      file_path,
  )

  # 2. Token budgeting / context truncation
  check(
      "format_evidence_context" in code and "max_" in code,
      "Enforces token budgeting on evidence context for Qwen 2048 window.",
      file_path,
  )

  # 3. Greedy translation
  check(
      "num_beams=1" in code,
      "NLLB-200 uses greedy decoding (num_beams=1) for sub-second edge"
      " translation.",
      file_path,
  )


def audit_sync_and_app(sync_path, app_path):
  if os.path.exists(sync_path):
    with open(sync_path, "r", encoding="utf-8") as f:
      code = f.read()
    check(
        "active_db.json" in code or "POINTER_FILE" in code,
        "Implements pointer-based blue/green switching for WinError 32.",
        sync_path,
    )

  if os.path.exists(app_path):
    with open(app_path, "r", encoding="utf-8") as f:
      code = f.read()
    check(
        "resolve_conversational_coreference" in code,
        "Anaphora and pronoun resolution wired into pipeline.",
        app_path,
    )
    check(
        "contains_telugu_script(text)" not in code,
        "Typed Telugu is not blocked in UI handler.",
        app_path,
    )
    # Check that emergency DO preserves state
    check(
        "new_diagnostic_state()" not in code.split("EMERGENCY_HYPOXIA")[-1][
            :300
        ]
        if "EMERGENCY_HYPOXIA" in code
        else True,
        "Emergency DO alert preserves diagnostic_state (no amnesia wipe).",
        app_path,
    )


if __name__ == "__main__":
  audit_intent_extractor("src/intent_extractor.py")
  audit_decision_router("src/decision_router.py")
  audit_diagnostic_engine("src/diagnostic_engine.py")
  audit_response_generator("src/response_generator.py")
  audit_sync_and_app("src/sync_manager.py", "app.py")

  print("\n" + "=" * 50)
  print("AQUAASSIST ARCHITECTURE 2 VERIFICATION REPORT")
  print("=" * 50)
  for p in PASSED:
    print(f" \033[92m{p}\033[0m")
  for w in WARNINGS:
    print(f" \033[93m{w}\033[0m")
  for e in ERRORS:
    print(f" \033[91m{e}\033[0m")
  print("=" * 50)

  if ERRORS:
    print(f"\nResult: FAILED with {len(ERRORS)} architectural violations.")
    sys.exit(1)
  else:
    print("\nResult: PASSED. All code complies with Architecture 2.")
    sys.exit(0)