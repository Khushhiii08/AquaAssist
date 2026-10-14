import json
import os
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

DATASET_PATH = "./data/evaluation/rlaif_dataset.json"
OUTPUT_DIR = "./models/deberta_aquaculture_router"
BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5

LABEL_MAP = {
    "ABSTAIN": 0,    # Contradiction
    "ANSWER": 1,     # Entailment
    "CLARIFY": 2     # Neutral
}

def load_training_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing training dataset at {filepath}")
        
    with open(filepath, "r") as f:
        data = json.load(f)

    samples = []
    for item in data:
        premise = item.get("evidence", "")
        hypothesis = item.get("declarative_hypothesis", "")
        decision = item.get("expected_decision", "CLARIFY").upper()
        
        if decision in LABEL_MAP and premise and hypothesis:
            samples.append(InputExample(
                texts=[premise, hypothesis],
                label=LABEL_MAP[decision]
            ))
    return samples

def train():
    print(f"[*] Loading training data from {DATASET_PATH}...")
    samples = load_training_data(DATASET_PATH)
    
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)
    print(f"[*] Split: {len(train_samples)} training, {len(val_samples)} validation samples")

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing base cross-encoder on device: {device}")
    
    model = CrossEncoder(
        "cross-encoder/nli-deberta-v3-small", 
        num_labels=3, 
        device=device
    )

    warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)
    print(f"[*] Beginning fine-tuning ({NUM_EPOCHS} epochs, lr={LEARNING_RATE})...")

    model.fit(
        train_dataloader=train_dataloader,
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': LEARNING_RATE},
        output_path=OUTPUT_DIR,
        show_progress_bar=True
    )
    # CRITICAL: Force PyTorch to write the weights to the disk
    print(f"[*] Writing weights to {OUTPUT_DIR}...")
    model.save(OUTPUT_DIR)
    
    print(f"[+] Model fine-tuned successfully and saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()