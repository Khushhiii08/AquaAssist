---
title: AquaAssist
emoji: 🐟
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# AquaAssist: Neuro-Symbolic Edge Aquaculture Diagnostic System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AquaAssist** is an offline-first, vernacular neuro-symbolic clinical diagnostic and water-quality decision engine for aquaculture pond managers (finfish and brackishwater shrimp). It couples deterministic symbolic guardrails (safety gates, telemetry bounds, syntactic dependency trees, and directional NLI verification) with neural language and speech processing (Qwen 2.5 SLM, Whisper.cpp, and NLLB-200).

---

## System Architecture

AquaAssist replaces unconstrained LLM hallucinations with a **Dual-Path Neuro-Symbolic Pipeline**:

```text
                              [Farmer Input: Telugu / English Audio or Text]
                                                    │
                                                    ▼
                                    [Whisper.cpp / Anaphora Resolver]
                                                    │
                                                    ▼
                                      [Intent & Telemetry Extractor]
                                        (spaCy Dependency Parsing)
                                                    │
                      ┌─────────────────────────────┴─────────────────────────────┐
                      ▼                                                           ▼
            [Fast-Path Safety Gates]                                     [Slow-Path Diagnostic]
     - Tier 1: Biological Impossibility                                             │
     - Tier 2: Chemical Incompatibility (e.g., Bleach+KMnO4)                        ▼
     - Tier 3: Out-of-Domain / Temporal Bounds                         [ChromaDB Vector Retrieval]
     - Tier 4: Acute Hypoxia Short-Circuit (<50ms)                      (Species & Limnology Facets)
                      │                                                           │
                      │                                                           ▼
                      │                                            [DeBERTa-v3 NLI Verification]
                      │                                            (Declarative Hypothesis Test)
                      │                                                           │
                      │                                                           ▼
                      │                                            [Qwen 2.5 1.5B GGUF Synthesis]
                      │                                              (Strict Context Grounding)
                      │                                                           │
                      └─────────────────────────────┬─────────────────────────────┘
                                                    ▼
                                      [NLLB-200 Vernacular Engine]
                                   (Greedy Real-Time Telugu Synthesis)
                                                    │
                                                    ▼
                                      [Dual-Language Clinical Output]
```

### Key Pillars

1. **Deterministic Fast-Path (< 50ms):**
   - **Emergency Hypoxia Gate:** Immediately detects lethal dissolved oxygen levels ($< 3.0\text{ mg/L}$) and triggers immediate aeration and feed-stoppage alarms without invoking LLMs.
   - **State Persistence with Fresh Tracking:** Distinguishes between fresh, acute telemetry readings and historical conversational context, preventing redundant alarm loops during follow-up queries.
   - **Chemical Safety Intercept:** Prevents hazardous mixing of oxidizing and reducing chemicals (e.g., calcium hypochlorite with potassium permanganate or formalin).

2. **Grounded Slow-Path (ChromaDB + DeBERTa NLI):**
   - **Metadata & Topic Isolation:** Partitions candidate search space by target organism (`fish` vs. `shrimp`) and limnological domain, preventing cross-contamination between water-quality issues and parasitic chemotherapeutics.
   - **Directional NLI Gate:** Uses a fine-tuned DeBERTa Cross-Encoder to evaluate premise-hypothesis entailment against declarative propositions, eliminating question bias and abstaining if evidence is missing.

3. **Constrained SLM Inference & Translation:**
   - **Local Inference:** Runs a quantized `Qwen-2.5-1.5B-Instruct` model locally via `llama.cpp` with strict GBNF grammar constraints.
   - **Vernacular Pipeline:** Real-time Telugu translation via `facebook/nllb-200-distilled-600M` using greedy decoding and domain-specific lexicon enforcement.

---

## Directory Structure

AquaAssist/
├── app.py                      # Gradio multi-turn UI & orchestration pipeline
├── constraints.gbnf            # Strict GBNF grammar rules for SLM generation
├── requirements.txt            # Production dependencies
├── verify_architecture.py      # Automated pipeline test matrix runner
├── src/
│   ├── diagnostic_engine.py    # ChromaDB retrieval, NLI scoring, & telemetry health
│   ├── intent_extractor.py     # Syntactic dependency parsing & state tracking
│   ├── response_generator.py   # Token-budgeted Qwen SLM synthesis & NLLB translation
│   ├── decision_router.py      # Policy safety gates (chemical, temporal, out-of-domain)
│   ├── decision_response.py    # Deterministic bilingual policy responses
│   ├── whisper_transcriber.py  # Local Telugu/English speech transcription
│   └── sync_manager.py         # Remote release asset synchronization (OTA DB updates)
├── data/
│   └── chroma_db/              # Indexed vector database (1,971 semantic chunks)
└── models/                     # GGUF models & DeBERTa router checkpoints


---

## Getting Started

### 1. Prerequisites

- Python 3.10+ (macOS Apple Silicon MPS, Linux CUDA, or CPU)
- `ffmpeg` (for audio transcription pipeline)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```
### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Khushhiii08/AquaAssist.git
cd AquaAssist

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

```

### 3. Knowledge Base & Models Setup
AquaAssist runs off verified ICAR-CIBA aquaculture manuals. When running app.py for the first time, sync_manager.py downloads the pre-indexed chroma_db.zip release bundle automatically if local vectors are not present.

To manually ingest raw documentation:
```bash
python src/ingest.py

```

### 4. Run Locally
Launch the Gradio interface on http://localhost:7860:
```bash
python app.py

```
