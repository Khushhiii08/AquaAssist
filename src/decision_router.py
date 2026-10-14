import torch
from sentence_transformers import CrossEncoder


MODEL_PATH = "./models/deberta_aquaculture_router"

ENTAILMENT_THRESHOLD = 0.75
NEUTRAL_THRESHOLD = 0.40
CONTRADICTION_THRESHOLD = 0.30


def load_model():
    """Load the fine-tuned DeBERTa NLI model."""
    return CrossEncoder(MODEL_PATH)


def classify_pair(model, evidence, hypothesis):
    """
    Classify an evidence-hypothesis pair.

    NLI order:
        evidence -> premise
        hypothesis -> hypothesis
    """

    logits = model.predict([[evidence, hypothesis]])

    probabilities = torch.softmax(
        torch.tensor(logits[0]),
        dim=0
    ).tolist()

    contradiction = float(probabilities[0])
    entailment = float(probabilities[1])
    neutral = float(probabilities[2])

    return {
        "contradiction": contradiction,
        "entailment": entailment,
        "neutral": neutral,
    }


def make_decision(probabilities):
    """Convert NLI probabilities into ANSWER, CLARIFY, or ABSTAIN."""

    contradiction = probabilities["contradiction"]
    entailment = probabilities["entailment"]
    neutral = probabilities["neutral"]

    if contradiction > CONTRADICTION_THRESHOLD:
        return "ABSTAIN"

    if entailment >= ENTAILMENT_THRESHOLD:
        return "ANSWER"

    if neutral > NEUTRAL_THRESHOLD:
        return "CLARIFY"

    return "CLARIFY"


if __name__ == "__main__":
    model = load_model()

    evidence = "The pond has adequate dissolved oxygen."
    hypothesis = "The pond has adequate dissolved oxygen."

    probabilities = classify_pair(
        model,
        evidence,
        hypothesis
    )

    decision = make_decision(probabilities)

    print("\nNLI Probabilities:")
    print(f"  Contradiction: {probabilities['contradiction']:.6f}")
    print(f"  Entailment:    {probabilities['entailment']:.6f}")
    print(f"  Neutral:       {probabilities['neutral']:.6f}")

    print(f"\nDecision: {decision}")