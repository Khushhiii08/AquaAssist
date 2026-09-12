import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# AquaAssist - Safety Coverage Visualization
# ============================================================

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Results from the verified invariant test suite.
categories = [
    "Valid values",
    "Low pH",
    "High pH",
    "Negative DO",
    "Boundary pH"
]

detected = [1, 1, 1, 1, 1]
total = [1, 1, 1, 1, 1]

coverage = [
    detected[i] / total[i] * 100
    for i in range(len(categories))
]


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(categories, coverage)

plt.ylabel("Safety detection coverage (%)")
plt.xlabel("Invariant test category")
plt.title("AquaAssist Biological Safety Guardrail Coverage")

plt.ylim(0, 110)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# Add percentage labels.
for i, value in enumerate(coverage):
    plt.text(
        i,
        value + 2,
        f"{value:.0f}%",
        ha="center"
    )

plt.tight_layout()

output_file = (
    OUTPUT_DIR /
    "phase2_safety_coverage.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Safety coverage figure saved to: {output_file}"
)
