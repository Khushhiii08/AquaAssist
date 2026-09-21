import os
import subprocess
import json
import platform
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 1. Multi-OS CLI Resolver
def get_llama_cli():
    if platform.system() == "Windows":
        # Meghana's local Windows path
        meghana_path = (
            r"C:\Users\tanag\AppData\Local\Microsoft\WinGet\Packages"
            r"\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-cli.exe"
        )
        if os.path.exists(meghana_path):
            return meghana_path
        return shutil.which("llama-cli.exe") or "llama-cli.exe"
    else:
        # Khushi's Mac/Linux path
        return shutil.which("llama-cli") or "llama-cli"

LLAMA_CLI = get_llama_cli()

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "qwen-2.5-1.5b-instruct.gguf"
)

GRAMMAR_PATH = os.path.join(
    PROJECT_ROOT,
    "constraints.gbnf"
)

PROMPT_CACHE_PATH = os.path.join(
    PROJECT_ROOT, 
    "prompt.cache"
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
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", "256",
        "-ngl", "99",                         # 2. GPU Offload: Forces max layers to Metal/CUDA
        "--no-display-prompt",
        "--grammar-file", GRAMMAR_PATH,
        "--prompt-cache", PROMPT_CACHE_PATH, 
        "--temp", "0"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    output = result.stdout or ""

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

        best_evidence = max(
            evidence,
            key=lambda item: item.get("entailment", 0.0)
        )

        data["evidence_chunk_id"] = best_evidence["id"]

        telugu = str(
            data.get("recommended_action_telugu", "")
        )

        telugu_chars = sum(
            1
            for ch in telugu
            if "\u0C00" <= ch <= "\u0C7F"
        )

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
        }
    ]

    result = generate_answer(
        "My shrimp are not eating and the pond water has become dark green.",
        test_evidence
    )

    print(result)