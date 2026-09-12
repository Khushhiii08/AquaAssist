import json
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path("data/evaluation/latency_results.json")
OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

labels = [
    "Model loading",
    "Embedding",
    "Retrieval"
]

values = [
    data["model_load_seconds"],
    data["embedding_mean_seconds"],
    data["retrieval_mean_seconds"]
]

plt.figure(figsize=(9, 6))
plt.bar(labels, values)

plt.ylabel("Time (seconds)")
plt.xlabel("Pipeline stage")
plt.title("AquaAssist Retrieval Pipeline Latency Breakdown")

for i, value in enumerate(values):
    plt.text(
        i,
        value + 0.1,
        f"{value:.4f} s",
        ha="center"
    )

plt.tight_layout()

output_file = OUTPUT_DIR / "phase2_latency_breakdown.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Latency figure saved to: {output_file}")
