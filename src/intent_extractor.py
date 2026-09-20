import os
import subprocess


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "qwen2.5-0.5b",
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

LLAMA_CLI = (
    r"C:\Users\tanag\AppData\Local\Microsoft\WinGet\Packages"
    r"\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\llama-cli.exe"
)


def extract_hypothesis(farmer_query):
    """Convert conversational farmer input into a concise diagnostic hypothesis."""

    prompt = f"""You are a strict information-preservation system.

Your task is ONLY to rewrite the farmer's statement as ONE short declarative observation.

CRITICAL RULES:
1. Copy every observed fact from the farmer.
2. Do NOT add any fact that is not explicitly stated.
3. Do NOT infer causes, diagnoses, reasons, treatments, or explanations.
4. Do NOT connect two observations with a causal relationship.
5. "not eating" means ONLY "not eating". It does NOT mean "lack of food" or "insufficient food".
6. "water became dark green" means ONLY that the water became dark green.
7. Preserve species, symptoms, colors, measurements, behaviors, and conditions exactly.
8. Do not give advice.
9. Do not explain anything.
10. Return ONLY ONE declarative sentence.

Example:
Farmer: My shrimp are not eating and the pond water became dark green.
Correct: My shrimp are not eating and the pond water has become dark green.
Wrong: My shrimp are not eating due to insufficient food.
Wrong: The shrimp are not eating because there is not enough food.
Wrong: The dark green water is causing the shrimp to stop eating.

Farmer statement:
{farmer_query}

Output only the rewritten observation:"""

    command = [
        LLAMA_CLI,
        "-m",
        MODEL_PATH,
        "-p",
        prompt,
        "-n",
        "40",
        "--no-display-prompt",
        "--single-turn"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Qwen inference failed:\n{result.stderr}"
        )

    output = result.stdout.strip()

    # llama.cpp may include its interactive banner and prompt.
    # Keep only the final generated response.
    lines = output.splitlines()

    clean_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("Loading model"):
            continue

        if line.startswith("▄▄") or line.startswith("██"):
            continue

        if line.startswith("build"):
            continue

        if line.startswith("model"):
            continue

        if line.startswith("ftype"):
            continue

        if line.startswith("modalities"):
            continue

        if line.startswith("available commands"):
            continue

        if line.startswith("/exit") or line.startswith("/regen"):
            continue

        if line.startswith("/clear") or line.startswith("/read"):
            continue

        if line.startswith("/glob"):
            continue

        if line.startswith(">"):
            continue

        if line.startswith("Rewrite the farmer"):
            continue

        if line.startswith("STRICT RULES"):
            continue

        clean_lines.append(line)

    # Remove llama.cpp performance information and exit message.
    final_lines = []

    for line in clean_lines:
        if line.startswith("[ Prompt:"):
            break

        if line == "Exiting...":
            break

        final_lines.append(line)

    if not final_lines:
        raise RuntimeError(
            "Could not extract hypothesis from llama.cpp output."
        )

    return final_lines[-1]


if __name__ == "__main__":

    farmer_query = (
        "My shrimp are not eating and the pond water became dark green."
    )

    hypothesis = extract_hypothesis(farmer_query)

    print("\nFarmer input:")
    print(farmer_query)

    print("\nExtracted hypothesis:")
    print(hypothesis)

