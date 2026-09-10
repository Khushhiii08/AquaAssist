# Evidence-Aware Vernacular RAG (EAV-RAG)

## Abstract

AI adoption in high-stakes agricultural domains such as aquaculture is constrained by unreliable internet connectivity and the vulnerability of edge-deployed Small Language Models (SLMs) to hallucination. This repository contains the source code for an Evidence-Aware Vernacular RAG (EAV-RAG) architecture designed to operate offline on standard edge hardware.

The system uses multilingual retrieval, local ChromaDB storage, and an NLI Cross-Encoder verification gate to improve evidence grounding for cross-lingual Telugu-English queries. Phase 2 extends the pipeline with deterministic three-way decision routing, automated RLAIF data generation, biological invariant verification, and constrained edge generation.

## Repository Architecture

- `src/`: Core Python scripts for ingestion, embedding, retrieval, and evidence verification.

- `data/`: Data pipeline directory.
  - `processed/`: Contains tracked processed JSON chunks.
  - `chroma_db/`: Local ChromaDB vector store; ignored by version control.
  - Raw PDFs are stored locally and ignored by version control.

- `docs/`: Engineering documentation, literature references, architecture diagrams, and academic materials.

## Initial Setup & Installation

Contributors should execute the following protocol to synchronize their local environments.

### 1. Clone the Repository

```bash
git clone https://github.com/Khushhiii08/AquaAssist.git
cd AquaAssist
```

### 2. Initialize Virtual Environment

```bash
python -m venv venv
```

On Windows:

```bash
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

## Phase 1 — Knowledge Base Construction

Phase 1 establishes the local aquaculture knowledge base from approved English aquaculture manuals and reports.

### Ingestion

`src/ingest.py`

- Extracts text from PDF documents using PyMuPDF.
- Generates fixed-size 150-word chunks with 30-word overlap.
- Applies rule-based metadata tagging for species and knowledge topics.
- Stores the processed chunks in `data/processed/chunks.json`.

### Embedding & Vector Storage

`src/embed.py`

- Uses the `BAAI/bge-m3` multilingual embedding model.
- Generates normalized dense vector representations.
- Stores embeddings in a persistent local ChromaDB collection.

### Retrieval Testing

`src/test_retrieval.py`

- Tests retrieval quality using benchmark queries.
- Records retrieved evidence and similarity scores for evaluation.

## Execution Pipeline: Phase 1 (Corpus Curation)

The initial phase focuses on the ingestion and processing of official ICAR-CIBA aquaculture manuals.

- `src/ingest.py`: Parses raw PDFs, generates fixed-size 150-word chunks with 30-word overlap, and appends rule-based metadata.

- `src/embed.py`: Utilizes the BGE-M3 multilingual model to convert text chunks into dense vectors, storing them in a local ChromaDB instance.

- `src/test_retrieval.py`: Tests the retrieval pipeline using benchmark queries and records the retrieved evidence and similarity scores.

## Phase 2 — Evidence Verification & Edge Generation

Phase 2 extends the Phase 1 retrieval pipeline with evidence verification, deterministic decision routing, automated RLAIF data generation, and constrained edge generation.

### NLI Cross-Encoder Evidence Gate

`src/cross_encoder_gate.py`

The current evidence verification pipeline:

1. Encodes the user query using BGE-M3.
2. Retrieves the top-15 candidate chunks from ChromaDB.
3. Evaluates query-evidence pairs using `cross-encoder/nli-deberta-v3-small`.
4. Filters evidence using the current entailment threshold of `0.85`.
5. Passes verified evidence chunks toward the local SLM generation stage.

### Three-Way Decision Policy

Phase 2 upgrades binary evidence filtering into a deterministic three-way routing policy:

- **ANSWER** — sufficient entailment evidence.
- **CLARIFY** — insufficiently supported or neutral evidence requiring clarification.
- **ABSTAIN** — contradictory or unsafe evidence.

Initial decision thresholds are subject to empirical calibration.

### Automated RLAIF Data Engine

Phase 2 includes an automated data-generation pipeline designed to synthesize cross-lingual NLI triplets:

- Entailment
- Neutral
- Contradiction

The generated samples are subjected to biological invariant verification before admission into the evaluation/training dataset.

### Constrained Edge Generation

Phase 2 also introduces:

- Quantized Qwen-2.5-1.5B-Instruct
- llama.cpp-based local generation
- GBNF-constrained structured output
- Offline edge execution
- Telugu response generation

These components are currently part of the ongoing Phase 2 development and integration process.

## Documentation & Figures

The `docs/` directory contains project documentation and publication-ready architecture diagrams.

Current Phase 2 diagrams include:

1. Two-Stage Cascaded Retrieval & Evidence Verification
2. Three-Way Evidence-Gated Decision State Machine
3. Automated RLAIF Data Generation & Verification Engine

## Current Development Status

### Phase 1

- Knowledge corpus curation — Completed
- PDF ingestion — Completed
- Chunk generation — Completed
- Metadata tagging — Completed
- BGE-M3 embedding — Completed
- ChromaDB storage — Completed
- Retrieval testing — Completed
- Phase 1 documentation — Completed

### Phase 2

- NLI Cross-Encoder evidence gate — Implemented
- Three-way decision policy — In development
- Automated RLAIF data engine — In development
- Biological invariant verification — In development
- GBNF-constrained generation — In development
- Qwen-2.5-1.5B edge generation — In development
- Streamlit edge interface — In development

## Project Contributors

- **Khushi** — Core ML Architecture & Pipeline Implementation
- **Meghana** — Integration, SLM Generation & UI
- **Jyothish** — Documentation, Logbook, Reference Management & Visual Architecture
- **Praisy** — Data Engine, Verification & Evaluation Support