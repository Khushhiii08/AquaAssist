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

**AquaAssist** is an offline-first, bilingual (Telugu & English) neuro-symbolic clinical diagnostic and water-quality decision support system for finfish and brackishwater shrimp pond managers. Built on **Architecture 2** (production-hardened anti-hallucination design), it pairs deterministic symbolic guardrails—multi-tier chemical/pharmacological policy gates, syntactic telemetry dependency parsing, aqueous ammonia equilibrium physics, and directional NLI verification—with edge-optimized neural models (`BAAI/bge-m3`, fine-tuned `DeBERTa-v3`, `Qwen 2.5 1.5B GGUF`, `NLLB-200`, and `Whisper`).

---

## System Architecture

AquaAssist eliminates unconstrained LLM hallucinations through a **Dual-Path Neuro-Symbolic Pipeline** that routes every query into one of three calibrated clinical actions (`ANSWER`, `CLARIFY`, or `ABSTAIN`):

```text
                [Farmer Input: Telugu / Tenglish / English Text or Voice]
                                           │
                                           ▼
                      [FFmpeg 16 kHz Mono + OpenAI Whisper (base)]
                       (Translates vernacular speech to English)
                                           │
                                           ▼
                       [Conversational Coreference & State Resolver]
                        (Multi-turn pronoun & slot-fill resolution)
                                           │
               ┌───────────────────────────┴───────────────────────────┐
               ▼                                                       ▼
   [Fast-Path Symbolic Gates (<50ms)]                     [Slow-Path Grounded RAG]
   • Tier 1: Domain & Biological Absurdity Guard                       │
   • Tier 2: Banned Drugs, Antibiotic Rx,                              ▼
             Folk Remedies & Chemical Mixing             [Intent & Telemetry Extractor]
   • Tier 3: Live Market, Feed Brand & Loan Filter      (spaCy Dependency Tree + Units)
   • Tier 4: Vague Complaint & Missing Context                         │
   • Telemetry Health & Conflict Reflexes:                             ▼
     - Multi-Sensor Conflict vs. Diurnal Cycle          [ChromaDB Faceted Retrieval]
     - Physical Bounds (DO, pH, Temp, TAN, ppt)         (BAAI/bge-m3 • 1,971 Chunks)
     - Acute Hypoxia Siren (DO < 3.0 mg/L)                             │
     - Emerson Toxic NH₃ Reflex (>= 0.05 mg/L)                         ▼
     - Optimal DO Fast Confirmation (>= 5.0 mg/L)       [DeBERTa-v3 Cross-Encoder NLI]
               │                                        (Declarative Hypothesis Gate)
               │                                                       │
               │                                                       ▼
               │                                        [Qwen 2.5 1.5B GGUF Synthesis]
               │                                        (Token-Budgeted + Guardrails)
               │                                                       │
               │                                                       ▼
               │                                        [Offline NLLB-200 Translator]
               │                                        (Unit-Shielded Telugu + Lexicon)
               └───────────────────────────┬───────────────────────────┘
                                           ▼
                       [Bilingual Advisory + Edge-TTS Voice + Log]
                        (te-IN-ShrutiNeural / en-IN-NeerjaNeural)
```

---

## What Happens Inside the Pipeline

### 1. Multimodal Input & Multi-Turn State Management (`app.py`, `src/whisper_transcriber.py`)
- **Voice & Vernacular Input:** Standardizes browser/uploaded audio to 16 kHz mono PCM WAV via `ffmpeg` and translates spoken Telugu or English directly to English text using local OpenAI Whisper (`base`). Typed English, Telugu Unicode (`తెలుగు`), and Romanized Telugu (`Tenglish`, e.g., *"royyalu baagolevu"*, *"cheruvulo DO taggindi"*) are accepted natively.
- **Coreference & Slot-Filling:** Resolves elliptical follow-up turns (*"what now?"*, *"how to fix that?"*) against persistent per-session diagnostic state (`species`, `habitat`, `telemetry`, `symptoms`, `pending_metrics`) without corrupting fresh numerical sensor updates.
- **Spoken Output & Audit Reports:** Generates MD5-cached spoken responses via `edge-tts` (prioritizing Telugu neural voice `te-IN-ShrutiNeural`) and downloadable diagnostic consultation reports (`AquaAssist_Diagnostic_Report.txt`), alongside an inspectable UI accordion showing live NLI entailment/contradiction traces.

### 2. Multi-Tier Policy & Chemical Safety Router (`src/decision_router.py`, `src/decision_response.py`)
Evaluates raw user queries before LLM invocation:
- **Tier 1 — Domain & Absurdity Filter:** Verifies aquaculture membership across lemmatized English, Telugu script, and Romanized Telugu inflections while rejecting terrestrial agronomy (wheat, paddy, crop fertilizers) and biological nonsense (*"why are shrimps not flying?"*).
- **Tier 2 — Pharmacological, Chemical & Scope Guardrails:**
  - **Banned Substances (`ABSTAIN`):** Intercepts prohibited aquaculture antibiotics and pesticides (chloramphenicol, nitrofurans, furazolidone, malachite green, endosulfan, dichlorvos/Nuvan, cypermethrin, sodium metabisulfite).
  - **Prescription & Biocide Dosing (`ABSTAIN`):** Refuses antibiotic/oxytetracycline prescriptions and uncalibrated therapeutic dosing of formalin, copper sulfate, or potassium permanganate (`KMnO₄`).
  - **Chemical Mixing & Stocked-Pond Hazards (`ABSTAIN`):** Blocks violent oxidizing/reducing combinations (e.g., bleaching powder + `KMnO₄`, formalin, ammonia, or probiotics) and direct bleaching powder disinfection in stocked ponds.
  - **Blind Conditioner Dosing (`CLARIFY`):** Withholds lime, zeolite, dolomite, or mineral dosages when pond area/depth/volume is missing or active mortality is occurring.
  - **Unproven Remedies, Evasion & Out-of-Scope Systems (`ABSTAIN`):** Rejects folk cure claims (turmeric/pasupu, neem leaves, common salt curing EHP/all diseases), residue-test evasion or hiding diseased shrimp, yield guarantees, and out-of-corpus culture systems (ornamental goldfish, scampi, freshwater cage culture).
- **Tier 3 — Offline Temporal & Commercial Boundary (`ABSTAIN`):** Intercepts live village/mandi shrimp pricing, feed/hatchery brand endorsements, bank loans, subsidies, and insurance queries.
- **Tier 4 — Vague Diagnostic Intercept (`CLARIFY`):** Requests water appearance, stock behavior, and test kit readings when complaints lack measurable symptoms or parameters (including TAN toxicity questions missing `pH` and temperature).

### 3. Hardened Telemetry Extraction & Limnological Reflexes (`src/intent_extractor.py`, `src/diagnostic_engine.py`)
- **Syntactic Dependency & Bijective Unit Parsing:** Navigates `spaCy` dependency trees to bind values to parameters (`DO`, `pH`, `TAN`, `NO2`, `NO3`, `alkalinity`, `temperature`, `salinity`, `transparency`, `depth`, `ORP`) while enforcing bijective physical unit invariance (`ppt`/`psu` → `salinity`, `°C` → `temperature`) and decoupling hardware words (`probe`, `meter`, `kit`).
- **Operational & Clock Noise Filtering:** Filters out timestamps (`5:30`, `4:45am`, `gantalaki`) and farm operational metadata (`DOC 78`, `4 trays`, `3 acre`, `pond 4`, `count 40`) so farm logs never overwrite water-quality readings.
- **Diurnal Epoch Indexing vs. Sensor Conflicts:** Indexes temporal markers (`morning`, `afternoon`, `dawn`, `night`) per reading. Concurrent diverging readings (`DO` spread `>= 2.0 mg/L` or `pH` spread `>= 1.5`) trigger sensor recalibration (`CONFLICT_READINGS`), whereas morning-to-afternoon swings are recognized as phytoplankton photosynthesis/respiration cycles.
- **Biological Limits & Emergency Short-Circuits:**
  - Rejects physically impossible readings (`DO` outside `0–22 mg/L`, `pH` outside `3.5–11.5`, `temperature` outside `5–45 °C`, `salinity` outside `0–60 ppt`).
  - **Acute Hypoxia (`DO < 3.0 mg/L`):** Fires an immediate bilingual emergency aeration and feed-stoppage alarm on fresh-turn readings (`< 50 ms`), while flagging unexplained marginal `DO` (`3.0–5.0 mg/L`) for clarification.
  - **Emerson Un-ionized Ammonia (`NH₃`) Equilibrium:** Computes toxic un-ionized `NH₃` from `TAN`, `pH`, and temperature; alerts when `NH₃ >= 0.05 mg/L` and warns against applying lime at high `pH`.
  - **Optimal DO Fast-Path (`DO >= 5.0 mg/L`):** Confirms safe oxygenation immediately when no clinical symptoms are present.

### 4. Faceted Retrieval, Directional NLI & Grounded Generation (`src/diagnostic_engine.py`, `src/response_generator.py`)
- **Knowledge Base:** **1,971 semantic chunks** distilled from **23 verified ICAR-CIBA, FAO, and brackishwater/finfish manuals** (`data/raw_pdfs/`), embedded with `BAAI/bge-m3` in ChromaDB and partitioned by `species` (`fish`, `shrimp`, `general`), `habitat`, and `primary_topic` (with larval hatchery `artemia`/`nauplii` exclusion during grow-out pond checks).
- **Declarative DeBERTa NLI Gate:** Rewrites farmer questions into declarative clinical propositions to eliminate question-neutrality bias, scoring `[evidence_chunk, declarative_hypothesis]` with a fine-tuned `DeBERTa-v3-small` Cross-Encoder (`models/deberta_aquaculture_router`). Contradicted claims (e.g., human paracetamol tablets for fish fungus) are strictly abstained.
- **Constrained SLM & Offline Telugu Synthesis:** Synthesizes concise operational steps using quantized `Qwen-2.5-1.5B-Instruct` (`llama-cpp-python`, 2048 context window, token-budgeted 3-chunk context, domain directives for viral WSSV non-antibiotic biosecurity, *P. vannamei* optimal vs. extreme salinity, diurnal `pH` gypsum guidance, polarity contradiction sanitization, and extractive prose fallback if GGUF weights are absent). Translates offline via `facebook/nllb-200-distilled-600M` using greedy decoding (`num_beams=1`), unit shielding (`ppt`, `mg/L`, `ppm`), and Telugu aquaculture calque correction.
- **Atomic Blue/Green OTA Sync (`src/sync_manager.py`):** Bootstraps missing ChromaDB and DeBERTa release bundles on startup and supports one-click sidebar knowledge updates using versioned directories and atomic `data/active_db.json` pointer swaps (preventing file-locking errors).

---

## Evaluation & Stress-Test Benchmarks

AquaAssist includes four automated verification and evaluation harnesses tracking architectural invariants and real-world messy farmer queries:

| Suite | Runner | Dataset / Scope | Current Result |
| :--- | :--- | :--- | :--- |
| **Architecture 2 Static Audit** | `testing/verify_architecture.py` | AST & pattern audit across `src/` and `app.py` (anti-hardcoding, real NLI, state & OTA invariants) | **PASSED (0 violations)** |
| **Formal Clinical & Safety Suite** | `testing/test_rigorous_eval_full.py` | 20 multi-turn & edge scenarios (`T1`–`T20`: hypoxia, chemical hazards, Emerson `NH₃`, diurnal `pH`, WSSV, coreference, OTA) | **20 / 20 (100.0%)** |
| **Calibrated Blind Benchmark** | `testing/run_blind_evaluation.py` | 258 queries across 4 member workbooks (`AQ-001`–`AQ-258` in `messy_query_excelsheets/`) | **248 / 258 (96.12%)** decision accuracy (~189 ms/query) |
| **End-to-End Generative Batch Stress Test** | `batch_evaluate_aquaassist.py` | Full SLM + NLLB execution on `messy_query_excelsheets/` (258) & noisy code-mixed `real_life_messy_queries/` (260), auto-filling Excel & `.jsonl` traces | Archived in `24OCT_real_life_query_test_results/` & `25OCT_after-stresstest-correction_results/` |

---

## Directory Structure

```text
AquaAssist/
├── app.py                              # Gradio multi-session UI, edge-tts audio, & unified pipeline
├── batch_evaluate_aquaassist.py        # Multi-workbook end-to-end evaluator (auto-fills Excel + JSONL traces)
├── constraints.gbnf                    # Strict GBNF JSON grammar used during offline manual ingestion
├── requirements.txt                    # Python dependencies (including pinned en_core_web_sm wheel)
├── packages.txt                        # OS-level packages for Hugging Face Spaces (ffmpeg, build-essential)
├── Dockerfile                          # Container build (Python 3.11-slim + pre-cached Whisper base)
├── docker-compose.yml                  # Local container stack + Cloudflare Quick Tunnel (trycloudflare)
├── src/
│   ├── intent_extractor.py             # spaCy dependency telemetry parser, noise filter, & diurnal tracker
│   ├── decision_router.py              # 4-tier deterministic domain, chemical, market, & vagueness router
│   ├── decision_response.py            # Authoritative bilingual (English & Telugu) policy advisories
│   ├── diagnostic_engine.py            # ChromaDB faceted retrieval, Emerson NH₃ math, & DeBERTa NLI gate
│   ├── response_generator.py           # Token-budgeted Qwen 2.5 GGUF synthesis & offline NLLB translation
│   ├── whisper_transcriber.py          # FFmpeg 16 kHz resampler & OpenAI Whisper speech-to-English
│   ├── sync_manager.py                 # Atomic blue/green pointer-based OTA updater (ChromaDB & DeBERTa)
│   ├── ingest.py                       # PDF manual parser, triplet extractor, GBNF synthesizer, & indexer
│   ├── embed.py                        # Standalone BAAI/bge-m3 chunk embedder
│   ├── generate_rlaif_data.py          # 3-way NLI triplet synthesizer (Entailment / Neutral / Contradiction)
│   ├── train_router.py                 # Fine-tuning script for DeBERTa-v3 aquaculture Cross-Encoder
│   ├── cross_encoder_gate.py           # Standalone Cross-Encoder retrieval verification utility
│   ├── persona_renderer.py             # CLI bilingual field advisory formatter
│   ├── run_ablation.py                 # Baseline NLI routing ablation & confusion matrix evaluator
│   ├── measure_latency.py              # BGE-M3 + ChromaDB retrieval latency profiler
│   ├── profile_qwen_latency.py         # Qwen GGUF memory & token latency profiler
│   ├── plot_latency.py                 # Retrieval latency figure generator
│   └── plot_safety_coverage.py         # Safety invariant coverage figure generator
├── testing/
│   ├── verify_architecture.py          # Static Architecture 2 compliance auditor
│   ├── test_rigorous_eval_full.py      # Formal 20-scenario (T1–T20) clinical & safety benchmark
│   ├── run_blind_evaluation.py         # Calibrated 258-query blind routing & NLI evaluator
│   ├── test_rc01_telemetry.py          # Telemetry unit invariance & timestamp exclusion tests
│   ├── test_rc02_schema_ammonia.py     # ChromaDB topic schema & Emerson toxic ammonia tests
│   ├── test_rc03_rc04_router.py        # Tenglish stems, banned substances, & vague gate tests
│   ├── test_rc05_rc07_generator.py     # Extractive fallback, polarity sanitizer, & Telugu calque tests
│   ├── test_aquaculture_invariants.py  # Biological boundary invariant unit tests
│   ├── test_intent_extractor.py        # Intent & conflicting measurement unit tests
│   └── test_retrieval.py               # Direct ChromaDB retrieval sanity check
├── data/
│   ├── raw_pdfs/                       # 23 ICAR-CIBA, FAO, & aquaculture reference manuals
│   ├── processed/chunks.json           # Processed text chunks
│   ├── chroma_db/                      # Pre-indexed ChromaDB store (aqua_assist, 1,971 chunks)
│   ├── evaluation/                     # RLAIF dataset & latency profiling JSONs
│   └── version.json                    # Knowledge base release metadata
├── models/
│   ├── deberta_aquaculture_router/     # Fine-tuned 3-class DeBERTa-v3 NLI Cross-Encoder checkpoint
│   └── qwen-2.5-1.5b-instruct.gguf     # Local quantized Qwen 2.5 SLM weights
├── messy_query_excelsheets/            # 4 member blind evaluation workbooks (AQ-001 to AQ-258)
├── real_life_messy_queries/            # 260 real-world noisy English / Telugu / Tenglish field queries
├── 24OCT_real_life_query_test_results/ # Executed workbooks, JSONL traces, & scorecards (24 Oct run)
└── 25OCT_after-stresstest-correction_results/ # Post-hardening executed workbooks & scorecards (25 Oct run)
```

---

## Getting Started

### 1. Prerequisites
- **Python 3.10+** (supports macOS Apple Silicon MPS/Metal, Linux CUDA, or CPU; thread-bounded for 2-vCPU edge/Spaces tiers)
- **`ffmpeg`** (required for microphone/uploaded audio standardization)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg build-essential
```

### 2. Local Installation

```bash
git clone https://github.com/Khushhiii08/AquaAssist.git
cd AquaAssist

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```
*(Note: `requirements.txt` installs `en_core_web_sm` directly via wheel; if needed manually, run `python -m spacy download en_core_web_sm`.)*

### 3. Models & Knowledge Base Setup
- **Automatic OTA Bootstrap:** On first launch of `app.py`, `src/sync_manager.py` automatically downloads and extracts `chroma_db.zip` and `deberta_aquaculture_router.zip` from GitHub Releases if not already present locally.
- **Local SLM Weights:** Place `qwen-2.5-1.5b-instruct.gguf` inside `models/` for dynamic generative synthesis (if absent, `src/response_generator.py` automatically degrades gracefully to grounded extractive prose fallback).
- **Translation Cache:** Ensure `facebook/nllb-200-distilled-600M` and `BAAI/bge-m3` are cached in `~/.cache/huggingface` while online once prior to air-gapped field operation (`response_generator.py` enforces `local_files_only=True` at runtime).
- **Re-ingesting Raw Manuals (Optional):**
  ```bash
  python src/ingest.py
  ```

### 4. Running the Application

#### Option A: Direct Python (Gradio UI)
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

#### Option B: Docker Compose + Public HTTPS Tunnel
```bash
docker compose up --build -d

# View the public trycloudflare.com HTTPS URL:
docker logs aquaassist-tunnel
```

---

## Running Verification & Benchmarks

```bash
# 1. Static Architecture 2 compliance check
python testing/verify_architecture.py

# 2. Formal 20-scenario clinical, multi-turn & latency benchmark (T1–T20)
python testing/test_rigorous_eval_full.py

# 3. Fast calibrated 258-query blind routing & NLI benchmark
python testing/run_blind_evaluation.py

# 4. End-to-end pipeline batch evaluation on Excel sheets (auto-fills decisions & writes JSONL traces)
python batch_evaluate_aquaassist.py --folder messy_query_excelsheets
python batch_evaluate_aquaassist.py --folder real_life_messy_queries
```
