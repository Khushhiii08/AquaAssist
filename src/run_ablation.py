import os
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from decision_router import NLIDecisionRouter

# Configuration
DATASET_PATH = "./data/evaluation/rlaif_dataset.json" 
BATCH_LIMIT = 100 # Keeping this small for our initial baseline test

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[-] Cannot find Praisy's dataset at {path}. Check the folder structure.")
    with open(path, "r") as f:
        return json.load(f)

def run_baseline_sweep():
    print("[*] Initializing Phase 2 NLI Decision Router...")
    router = NLIDecisionRouter()
    
    print(f"[*] Loading synthetic dataset from {DATASET_PATH}...")
    try:
        dataset = load_dataset(DATASET_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Slicing the dataset for a faster local test
    test_batch = dataset[:BATCH_LIMIT]
    
    y_true = []
    y_pred = []
    
    print(f"[*] Executing Ablation Sweep on {len(test_batch)} samples...")
    for item in tqdm(test_batch, desc="Evaluating Thresholds"):
        # Mapping Praisy's schema correctly to prevent empty string evaluations
        extracted_hypothesis = item.get("declarative_hypothesis", "")
        retrieved_chunk = item.get("evidence", "")
        expected_decision = item.get("expected_decision", "CLARIFY").upper()
        
        # Feeding it through your newly updated DeBERTa evaluation gate
        decision, metrics = router.evaluate_routing(extracted_hypothesis, retrieved_chunk)
        
        y_true.append(expected_decision)
        y_pred.append(decision)

    print("\n" + "="*50)
    print(" BASELINE ZERO-SHOT PERFORMANCE (DOMAIN GAP)")
    print("="*50)
    
    # Generating IEEE-ready metrics
    labels = ["ANSWER", "CLARIFY", "ABSTAIN"]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"{labels}")
    for i, row in enumerate(cm):
        print(f"{labels[i].ljust(10)} {row}")
        
    print("\n[*] Baseline sweep complete.")

if __name__ == "__main__":
    run_baseline_sweep()