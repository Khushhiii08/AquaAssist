import os
import subprocess
import shutil
import json
import tempfile
import platform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "qwen-2.5-0.5b-instruct.gguf"
)

def get_llama_cli_path():
    """
    Dynamically locates the llama-cli executable across any OS.
    Prioritizes an environment variable, then searches the system PATH.
    """
    # 1. Check if an environment variable override is provided
    env_path = os.getenv("LLAMA_CLI_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. Automatically search system PATH across platforms
    # Note: shutil.which automatically checks Windows PATHEXT (.exe, .bat, etc.)
    cli_name = "llama-cli.exe" if platform.system() == "Windows" else "llama-cli"
    system_path = shutil.which(cli_name)
    if system_path:
        return system_path

    # 3. Fallback string if not found in PATH (will raise a clean OS error on execution)
    return "llama-cli"

LLAMA_CLI = get_llama_cli_path()

def extract_hypothesis(farmer_query):
    """Convert conversational farmer input into a rich, technically mapped hypothesis for high-precision vector retrieval."""

    prompt = f"""You are an expert aquaculture diagnostician and data normalizer.
Your job is to read a farmer's colloquial observation and map it to formal aquaculture symptoms and parameters.
- Translate everyday descriptions (e.g., "dark green water") into standard technical terminology (e.g., "dense phytoplankton bloom / high microalgae concentration").
- Translate behavioral cues (e.g., "not eating") into canonical terms (e.g., "anorexia / reduced feed intake").
- Output ONLY valid JSON matching this exact schema:
{{"symptoms": ["technical symptom 1", "technical symptom 2"], "parameters": {{"parameter_name": "value_or_status"}}, "clinical_hypothesis": "A concise, professional diagnostic sentence combining these findings for vector search."}}

Farmer statement:
{farmer_query}

JSON Output:
"""

    temp_in_path = None
    temp_out_path = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as temp_in:
            temp_in.write(prompt)
            temp_in_path = temp_in.name

        with tempfile.NamedTemporaryFile(mode="r", delete=False, suffix=".out", encoding="utf-8") as temp_out:
            temp_out_path = temp_out.name

        command = [
            LLAMA_CLI,
            "-m", MODEL_PATH,
            "-f", temp_in_path,
            "-n", "200",
            "--single-turn",
            "--no-display-prompt"
        ]

        with open(temp_out_path, "w", encoding="utf-8") as out_file:
            process_result = subprocess.run(
                command,
                stdout=out_file,
                stderr=subprocess.DEVNULL, 
                text=True,
                encoding="utf-8"
            )

            if process_result.returncode != 0:
                raise RuntimeError("Qwen inference execution failed.")

        with open(temp_out_path, "r", encoding="utf-8") as result_file:
            raw_output = result_file.read()

        start_idx = raw_output.find("{")
        end_idx = raw_output.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON brackets found in output.")
            
        clean_json_string = raw_output[start_idx:end_idx]
        data = json.loads(clean_json_string)
        
        # Fallback to raw text if model output is missing or generic
        clinical_hypothesis = data.get("clinical_hypothesis", "")
        if not clinical_hypothesis or "requires parameter verification" in clinical_hypothesis:
            return farmer_query
            
        return clinical_hypothesis
        
    except (json.JSONDecodeError, ValueError):
        return "The aquaculture system requires parameter verification for shrimp health and water quality."
    except Exception as e:
        print(f"[!] System Error during extraction: {e}")
        return "CLARIFY"
    finally:
        if temp_in_path and os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if temp_out_path and os.path.exists(temp_out_path):
            os.remove(temp_out_path)


if __name__ == "__main__":
    farmer_query = "My shrimp are swimming near the surface and gasping for air."
    hypothesis = extract_hypothesis(farmer_query)

    print("\nFarmer input:")
    print(farmer_query)

    print("\nExtracted hypothesis:")
    print(hypothesis)