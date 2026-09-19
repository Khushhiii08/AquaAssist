# Evidence-Aware Vernacular RAG (EAV-RAG)

## Abstract

AI adoption in high-stakes agricultural domains such as aquaculture is constrained by unreliable internet connectivity and the vulnerability of edge-deployed Small Language Models (SLMs) to hallucination.

This repository contains the source code for an **Evidence-Aware Vernacular RAG (EAV-RAG)** architecture designed to support offline aquaculture diagnostics on edge hardware.

The system uses multilingual BGE-M3 retrieval, local ChromaDB storage, NLI-based evidence verification, deterministic three-way decision routing, and locally deployed Qwen language models.

The project is being developed in multiple phases, covering knowledge-base construction, evidence verification, RLAIF data generation, safety routing, and constrained edge generation.

---

## Repository Architecture

```text
AquaAssist/
│
├── src/
│   ├── ingest.py
│   ├── embed.py
│   ├── test_retrieval.py
│   ├── cross_encoder_gate.py
│   ├── decision_router.py
│   ├── intent_extractor.py
│   ├── generate_rlaif_data.py
│   ├── test_aquaculture_invariants.py
│   ├── train_router.py
│   ├── measure_latency.py
│   └── profile_qwen_latency.py
│
├── data/
│   ├── processed/
│   │   └── chunks.json
│   ├── evaluation/
│   │   └── rlaif_dataset.json
│   └── chroma_db/
│
├── docs/
│   ├── figures/
│   └── engineering documentation
│
├── models/
│   └── Local model files
│
└── README.md
```

- `src/`: Core Python scripts for ingestion, retrieval, evidence verification, routing, data generation, and profiling.
- `data/processed/`: Processed aquaculture evidence chunks.
- `data/evaluation/`: Evaluation and RLAIF datasets.
- `data/chroma_db/`: Local ChromaDB vector store.
- `docs/`: Engineering documentation, references, and architecture diagrams.
- `models/`: Local model files. Model weights are excluded from version control.

---

## Initial Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Khushhiii08/AquaAssist.git
cd AquaAssist
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

On Windows:

```powershell
.\venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses components such as:

- Python
- PyMuPDF
- Sentence Transformers
- BGE-M3
- ChromaDB
- DeBERTa-based NLI models
- PyTorch
- llama.cpp
- Qwen instruction-tuned models

---

# Project Phases

## Phase 0 — Dataset Preparation and Evidence Correction

Phase 0 focused on improving the quality of the evidence used by the RLAIF dataset and NLI pipeline.

### Completed Work

- Updated the ingestion pipeline in `src/ingest.py`.
- Changed the evidence chunking process to produce focused chunks of at most **50 words**.
- Rebuilt the processed dataset in `data/processed/chunks.json`.
- Rebuilt the local ChromaDB collection using the corrected chunks.
- Regenerated `data/evaluation/rlaif_dataset.json`.
- Removed temporary and generated files that were not intended for version control.

### Validation Results

- Total processed chunks: **1,783**
- Maximum chunk size: **50 words**
- Chunks exceeding 50 words: **0**
- Total RLAIF samples: **5,349**
- Evidence samples exceeding 50 words: **0**
- `ANSWER` samples: **1,783**
- `CLARIFY` samples: **1,783**
- `ABSTAIN` samples: **1,783**
- ChromaDB collection size: **1,783 chunks**

---

## Phase 1 — Knowledge Base Construction

Phase 1 established the local aquaculture knowledge base using approved aquaculture manuals and reports.

### Ingestion

File: `src/ingest.py`

- Extracts text from PDF documents using PyMuPDF.
- Produces focused evidence chunks with a maximum length of 50 words.
- Applies rule-based metadata tagging for species and knowledge topics.
- Stores processed chunks in `data/processed/chunks.json`.

### Embedding and Vector Storage

File: `src/embed.py`

- Uses the `BAAI/bge-m3` multilingual embedding model.
- Generates normalized dense vector representations.
- Stores embeddings in a persistent local ChromaDB collection.

### Retrieval Testing

File: `src/test_retrieval.py`

- Tests retrieval using aquaculture benchmark queries.
- Records retrieved evidence and similarity scores.
- Supports evaluation of the local retrieval pipeline.

### Phase 1 Status

- Knowledge corpus curation — Completed
- PDF ingestion — Completed
- Evidence chunk generation — Completed
- Metadata tagging — Completed
- BGE-M3 embedding — Completed
- ChromaDB storage — Completed
- Retrieval testing — Completed
- Phase 1 documentation — Completed

---

## Phase 2 — Evidence Verification and Edge Generation

Phase 2 extends the retrieval pipeline with query extraction, NLI-based evidence verification, deterministic decision routing, RLAIF data generation, and local edge-model profiling.

## Phase 2 Architecture

The planned cascaded pipeline is:

```text
Farmer Query
     │
     ▼
Qwen-2.5-0.5B Pre-Retrieval Intent Extraction
     │
     ▼
BGE-M3 Query Embedding
     │
     ▼
ChromaDB Top-K Retrieval
     │
     ▼
DeBERTa NLI Evidence Verification
     │
     ▼
Three-Way Decision Router
 ┌──────────┬───────────┬───────────┐
 │ ANSWER   │ CLARIFY   │ ABSTAIN   │
 └──────────┴───────────┴───────────┘
     │
     ▼
Qwen-2.5-1.5B Local Generation
     │
     ▼
Constrained Response
```

The complete end-to-end integration is still under development.

### 1. Pre-Retrieval Intent Extraction

File: `src/intent_extractor.py`

The pre-retrieval extraction layer uses `qwen-2.5-0.5b-instruct.gguf` through `llama.cpp`.

Its purpose is to:

- Process conversational farmer queries.
- Remove greetings, questions, and unnecessary conversational content.
- Extract the main biological or chemical symptoms.
- Convert the input into a declarative hypothesis beginning with:

```text
The pond exhibits...
```

Initial local testing showed successful extraction for several test cases, although repetitive output was observed in one case and requires further refinement.

### 2. NLI Cross-Encoder Evidence Gate

File: `src/cross_encoder_gate.py`

The evidence-verification component currently:

1. Encodes the query using BGE-M3.
2. Retrieves the top-15 candidate chunks from ChromaDB.
3. Evaluates query-evidence pairs using `cross-encoder/nli-deberta-v3-small`.
4. Filters candidate evidence using the current entailment threshold of `0.85`.
5. Returns the verified evidence chunks.

The integration of the extracted hypothesis into the complete retrieval and verification pipeline is still in progress.

### 3. Three-Way Decision Policy

File: `src/decision_router.py`

The planned deterministic routing policy contains three possible decisions:

- **ANSWER** — the evidence sufficiently supports the hypothesis.
- **CLARIFY** — the available evidence is insufficient or neutral.
- **ABSTAIN** — contradictory or potentially unsafe evidence is detected.

The initial routing thresholds are:

- Entailment: `>= 0.75`
- Neutral: `> 0.40`
- Contradiction: `> 0.30`

These thresholds require further empirical evaluation and calibration.

### 4. Automated RLAIF Data Engine

The RLAIF data-generation pipeline produces synthetic NLI examples across three categories:

- Entailment
- Neutral
- Contradiction

The regenerated dataset currently contains:

- **5,349 total samples**
- **1,783 ANSWER samples**
- **1,783 CLARIFY samples**
- **1,783 ABSTAIN samples**

The dataset includes evidence, declarative hypotheses, NLI labels, and expected routing decisions.

Further validation is required to confirm that biological invariant checks are applied to every generated sample before it is added to the dataset.

### 5. Biological Invariant Verification

File: `src/test_aquaculture_invariants.py`

The project includes biological invariant verification intended to identify invalid or unsafe aquaculture-related claims.

The invariant-verification function has been implemented and tested separately. Complete integration with the RLAIF generation workflow requires further confirmation.

### 6. Local Edge Generation

The project includes local Qwen models for edge-based language generation:

- `qwen-2.5-0.5b-instruct.gguf`
- `qwen-2.5-1.5b-instruct.gguf`

The models are executed locally using `llama.cpp`.

Planned generation features include:

- Offline inference
- Local response generation
- GBNF-constrained structured output
- Vernacular response support

Full integration and GBNF generation verification are still pending.

### 7. Model Latency and Resource Profiling

File: `src/profile_qwen_latency.py`

Initial local profiling was performed for the two Qwen models.

| Model | Load Time | TTFT | Total Generation Time | Peak Memory |
|---|---:|---:|---:|---:|
| Qwen-2.5-0.5B | 0.56 s | 0.44 s | 3.75 s | 489.05 MB |
| Qwen-2.5-1.5B | 3.22 s | 1.06 s | 6.13 s | 1400.74 MB |

These measurements represent local model-level profiling. Full end-to-end pipeline profiling is still pending.

---

## Documentation and Figures

The `docs/` directory contains engineering documentation, literature references, and architecture diagrams.

Current Phase 2 diagrams include:

1. Cascaded Retrieval and Evidence Verification Pipeline
2. Three-Way Evidence-Gated Decision State Machine
3. Automated RLAIF Data Generation and Verification Engine
4. Phase 2 Latency Breakdown
5. Updated Phase 1 and Phase 2 Pipeline Diagrams

---

## Current Development Status

### Phase 0

- Dataset evidence correction — Completed
- 50-word evidence chunking — Completed
- RLAIF dataset regeneration — Completed
- Dataset label balancing — Completed
- ChromaDB reconstruction — Completed

### Phase 1

- Knowledge corpus curation — Completed
- PDF ingestion — Completed
- Evidence chunk generation — Completed
- Metadata tagging — Completed
- BGE-M3 embedding — Completed
- ChromaDB storage — Completed
- Retrieval testing — Completed
- Documentation — Completed

### Phase 2

- Pre-retrieval intent extraction — Implemented and locally tested
- NLI Cross-Encoder evidence gate — Implemented
- Three-way decision policy — Implemented, calibration pending
- RLAIF data generation — Implemented
- Biological invariant verification — Implemented separately; integration pending
- DeBERTa router fine-tuning — Pending
- GBNF-constrained generation — Pending verification
- Qwen-2.5-1.5B edge generation — Partially implemented
- Complete end-to-end integration — Pending
- End-to-end performance evaluation — Pending
- Streamlit interface — Planned/in development

---

## Project Contributors

- **Khushi** — Core ML Architecture and Pipeline Implementation
- **Meghana** — Integration, SLM Generation, and UI
- **Jyothish** — Documentation, Logbook, Reference Management, and Visual Architecture
- **Praisy** — Data Engine, Verification, and Evaluation Support