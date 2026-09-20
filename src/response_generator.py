import os
import subprocess
import json


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

LLAMA_CLI = (
    r"C:\Users\tanag\AppData\Local\Microsoft\WinGet\Packages"
    r"\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\llama-cli.exe"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "qwen2.5-1.5b",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)

GRAMMAR_PATH = os.path.join(
    PROJECT_ROOT,
    "constraints.gbnf"
)


def generate_answer(hypothesis, evidence):

    if not evidence:
        return json.dumps(
            {
                "evidence_chunk_id": "",
                "parameter_identified": "Unknown",
                "recommended_action_telugu":
                    "ఇచ్చిన ఆధారాలు అందుబాటులో లేవు."
            },
            ensure_ascii=False,
            indent=2
        )

    evidence_text = "\n".join(
        f"[{item['id']}] {item['text'][:500]}"
        for item in evidence
    )

    prompt = f"""You are a careful aquaculture diagnostic assistant.

Your job is to produce a short, evidence-grounded response for a farmer.

FARMER OBSERVATION:
{hypothesis}

VERIFIED EVIDENCE:
{evidence_text}

IMPORTANT SAFETY RULES:

1. Use ONLY information explicitly supported by the verified evidence.
2. Never invent a diagnosis, measurement, treatment, chemical, dosage, or procedure.
3. Never invent a recommended action.
4. If the evidence does NOT provide a specific action, clearly say so in simple Telugu.
5. The Telugu sentence must be natural and meaningful Telugu.
6. Do NOT output random Telugu words.
7. Do NOT output meaningless transliterations.
8. parameter_identified must be a short English description of the relevant aquaculture parameter.
9. evidence_chunk_id must be one of the provided evidence IDs.
10. Return ONLY the required JSON object.
11. Do not add explanations outside the JSON object.

If the evidence only describes a problem but does not provide a
specific corrective action, use this exact Telugu sentence:

"ఇచ్చిన ఆధారాల్లో నిర్దిష్ట చర్య సూచించబడలేదు."

The output must contain exactly these fields:

- evidence_chunk_id
- parameter_identified
- recommended_action_telugu
"""

    command = [
        LLAMA_CLI,
        "-m",
        MODEL_PATH,
        "-p",
        prompt,
        "-n",
        "100",
        "--no-display-prompt",
        "--single-turn",
        "--grammar-file",
        GRAMMAR_PATH,
        "--temp",
        "0"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    output = result.stdout or ""

    # Extract JSON from llama.cpp output.
    if "{" in output and "}" in output:
        output = output[
            output.find("{"):output.rfind("}") + 1
        ]

    output = output.strip()

    # ---------------------------------------------------------
    # Validate and sanitize model output
    # ---------------------------------------------------------

    try:
        data = json.loads(output)

        # Always select the strongest NLI-supported evidence.
        best_evidence = max(
            evidence,
            key=lambda item: item.get("entailment", 0.0)
        )

        data["evidence_chunk_id"] = best_evidence["id"]

        telugu = str(
            data.get("recommended_action_telugu", "")
        )

        # Count Telugu Unicode characters.
        telugu_chars = sum(
            1
            for ch in telugu
            if "\u0C00" <= ch <= "\u0C7F"
        )

        # Reject empty/corrupted Telugu output.
        if telugu_chars < 5:
            data["recommended_action_telugu"] = (
                "ఇచ్చిన ఆధారాల్లో నిర్దిష్ట చర్య సూచించబడలేదు."
            )

        if not data.get("parameter_identified"):
            data["parameter_identified"] = (
                "Aquaculture parameter"
            )

        return json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        )

    except (json.JSONDecodeError, TypeError, ValueError):

        # Deterministic safe fallback.
        best_evidence = max(
            evidence,
            key=lambda item: item.get("entailment", 0.0)
        )

        return json.dumps(
            {
                "evidence_chunk_id": best_evidence["id"],
                "parameter_identified":
                    "Aquaculture parameter",
                "recommended_action_telugu":
                    "ఇచ్చిన ఆధారాల్లో నిర్దిష్ట చర్య సూచించబడలేదు."
            },
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":

    test_evidence = [
        {
            "id": "CHK-00189",
            "text": (
                "Shrimp farms face disease outbreaks "
                "and water quality problems."
            ),
            "metadata": {},
            "distance": 0.8589,
            "entailment": 0.3259
        },
        {
            "id": "CHK-00465",
            "text": (
                "Deteriorating water quality has caused "
                "disease, mortality and slow growth of shrimp."
            ),
            "metadata": {},
            "distance": 0.8652,
            "entailment": 0.9982
        },
        {
            "id": "CHK-00505",
            "text": (
                "Stocking density affects shrimp growth, "
                "survival and yield."
            ),
            "metadata": {},
            "distance": 0.8930,
            "entailment": 0.9995
        }
    ]

    result = generate_answer(
        "My shrimp are not eating and the pond water has become dark green.",
        test_evidence
    )

    print(result)