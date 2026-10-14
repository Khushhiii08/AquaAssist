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

# AquaAssist 🦐🐟

**An Evidence-Grounded, Edge-Oriented Aquaculture Decision-Support System with Explicit Safety Handling**

---

## 📌 Overview

In rural aquaculture, farmers frequently encounter rapid water-quality shifts (such as critical dissolved oxygen depletion and pH fluctuations) and aquatic disease outbreaks under limited or unreliable internet connectivity. While conventional monitoring systems stop at raw sensor alerts and general-purpose large language models often hallucinate risky, ungrounded domain advice, **AquaAssist** delivers a local-first, evidence-grounded diagnostic workflow tailored for resource-constrained edge hardware.

AquaAssist separates offline knowledge-base preparation from an ultra-safe, deterministic farmer-facing diagnostic runtime. It implements a rigorous **Retrieve $\rightarrow$ Verify $\rightarrow$ Decide $\rightarrow$ Construct** pipeline that enforces biological boundaries, resolves conversational context, verifies candidate evidence via Natural Language Inference (NLI), and selectively routes actions into three explicit states: **ANSWER**, **CLARIFY**, or **ABSTAIN**.

---

## 🏗️ System Architecture

AquaAssist replaces unrestricted runtime generative text with deterministic extraction and cross-encoder hypothesis verification:

```
[Farmer Query (Voice / Text)]
            │
            ▼
[Whisper STT (16 kHz)] / Text Input
            │
            ▼
[intent_extractor.py] ─── (Regex + Entity Heuristics)
            │
            ├──► Critical Telemetry Check (DO < 3.0 ppm) ──► [EMERGENCY BYPASS]
            ├──► Biological Boundary Guardrails (pH 4.0–11.0, DO > 20.0 ppm)
            │
            ▼
[Candidate Retrieval] ─── BAAI/bge-m3 (1024-dim) + ChromaDB
            │
            ├──► Distance & Species Filtering (Distance > 0.80 rejected; species-aligned)
            │
            ▼
[Safety Verification] ─── DeBERTa-v3-small Cross-Encoder (NLI)
            │             (Entailment, Neutral, Contradiction)
            ▼
[Decision Routing Logic]
    ├── ANSWER   ── (Entailment >= 0.75, fully grounded by evidence)
    ├── CLARIFY  ── (Insufficient evidence or missing required parameters)
    └── ABSTAIN  ── (Contradiction >= 0.90, invalid telemetry, or unsupported advice)
            │
            ▼
[response_generator.py] ── Stored Expert Advice + Grounded Numerical Telemetry
            │
            ▼
[Multimodal Output] ───── Gradio Dashboard / edge-tts Audio Output (English & Telugu)
```

---

## ✨ Key Features

- **Local-First & Edge-Optimized**: Engineered to operate on a 4-core CPU with $\le 8\text{ GB}$ RAM, targeting a peak resident memory of $\le 2.5\text{ GB}$ and an end-to-end response time of $\le 7.0\text{ s}$. Supports Apple Silicon MPS, NVIDIA CUDA FP16, and multi-threaded CPU AVX2/NEON execution.
- **Deterministic Parameter & Intent Extraction**: `intent_extractor.py` handles species, symptoms, water-quality measurements, and conversational slot-filling without stochastic runtime LLMs. Includes an ultra-insensitive regular-expression schema built for field-facing human-machine interface (HMI) fault tolerance (e.g., handling variations like `D.0`, `d.o`, and irregular spacing).
- **Multi-Layered Safety Guardrails**:
  - **Biological Invariant Checks**: Intercepts impossible physical values (e.g., $\text{pH} < 4.0$ or $> 11.0$, $\text{DO} > 20.0\text{ mg/L}$) prior to vector-space search.
  - **Emergency Telemetry Bypass**: A parsed $\text{DO} < 3.0\text{ ppm}$ immediately triggers an emergency alert, bypassing neural retrieval and inference entirely.
  - **Contradiction Detection**: Halts normal diagnostic processing if conflicting telemetry is supplied (e.g., $\text{DO} = 2.5\text{ mg/L}$ and $8.0\text{ mg/L}$).
- **NLI Hypothesis Verification**: Does not treat vector similarity as diagnostic proof. A domain-adapted DeBERTa-v3-small Cross-Encoder verifies whether retrieved evidence entails ($\ge 0.75$) or contradicts ($\ge 0.90$) the diagnostic hypothesis.
- **Zero-Drift Response Construction**: `response_generator.py` binds pre-verified expert advice (curated from 23 aquaculture and veterinary reference manuals) with deterministic boilerplate and explicitly grounded numerical telemetry.
- **Bilingual & Multimodal Support**: Accommodates voice input (Whisper), English and Telugu text output (with offline NLLB-200 translation), and cached text-to-speech (edge-tts with LRU caching).

---

## 📊 Technical Stack & Subsystem Roles

| Component | Model / Configuration | Role in AquaAssist |
|---|---|---|
| **Semantic Embeddings** | `BAAI/bge-m3` (1024-dim) | Encodes normalized queries and knowledge base chunks |
| **Vector Database** | `ChromaDB 1.5.9` | Local vector storage (4,384 indexed documents) |
| **NLI Verification Gate** | `DeBERTa-v3-small` Cross-Encoder | Conditional Entailment / Neutral / Contradiction classification |
| **Query Extraction** | Python regex + token/entity heuristics | Deterministic intent, parameter extraction, and slot filling |
| **Response Construction** | `response_generator.py` | Selects stored advice and injects grounded numerical values |
| **Knowledge-Base Prep** | `Qwen-2.5-1.5B` (llama.cpp, GGUF + GBNF) | Offline structured extraction/synthesis (not used at runtime) |
| **Translation** | `NLLB-200` | Offline English-to-Telugu translation during corpus preparation |
| **Voice Transcription** | `Whisper` / `faster-whisper` | 16 kHz mono local speech-to-text |
| **User Interface** | `Gradio` | Field-facing interactive dashboard |

---

## 📈 Evaluation & Benchmark Summary

Experimental evaluation across targeted engineering scenarios (T1–T6) and a 20-case guardrail validation matrix demonstrates:

- **Retrieval Accuracy**: 80% direct relevance (4 of 5 top-ranked candidates) on surface emergence queries; 93.3% (14 of 15) manual evidence queries fully supported by the corpus.
- **Guardrail Conformance**: **20/20 (100%)** expected deterministic outcomes verified across critical low-DO, impossible telemetry, and out-of-domain input categories.
- **Conversational Slot Inheritance**: Successfully inherited standalone numerical follow-up values (e.g., inheriting `"2.0"` into a pending `DO` slot) and preserved temporal context isolation.
- **Component Latency**:
  - `BGE-M3` cold-start loading: $\approx 32.12\text{ s}$ (one-time initialization overhead)
  - Mean embedding time: $0.426\text{ s}$
  - Mean ChromaDB retrieval time: $0.014\text{ s}$
  - Total retrieval pipeline latency: $\mathbf{0.440\text{ s}}$

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Minimum hardware: 4-Core CPU, 8 GB RAM (Supports Apple Silicon MPS & x86-64 AVX2)
- Docker (optional, recommended for reproducible deployment)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/AquaAssist.git
   cd AquaAssist
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Knowledge Base & Vector Store:**
   Sync and populate the local ChromaDB database:
   ```bash
   python sync_manager.py --init
   ```

4. **Run the Gradio Interface:**
   ```bash
   python app.py   # serves on http://localhost:7860
   ```

---

## 🐳 Docker Deployment

AquaAssist can be run in a containerized environment, keeping database state decoupled from the container image for seamless over-the-air (OTA) atomic updates:

```bash
# Build the Docker image
docker build -t aqua-assist:latest .

# Run the container (the knowledge base is fetched via OTA from the GitHub release on first boot)
docker run -p 7860:7860 aqua-assist:latest
```

---

## 🔬 Known Limitations & Future Work

As analyzed in the technical evaluation:
- **Canonical Slot Normalization**: Addressing parameter mapping mismatches (e.g., standardizing `"dissolved oxygen"` directly to canonical `DO` to prevent state stalls).
- **Vernacular Speech Recognition**: Improving acoustic speech-to-text accuracy for diverse regional Telugu dialects.
- **Diagnostic Subject Persistence**: Enhancing multi-turn subject tracking to maintain specific species context (e.g., shrimp vs. pond water) across multi-turn follow-ups.

---
