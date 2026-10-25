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

**AquaAssist** is a local-first, bilingual (Telugu & English) neuro-symbolic clinical diagnostic and water-quality decision-support system for finfish and brackishwater shrimp pond management. Built on **Architecture 2** (production-hardened anti-hallucination design), it pairs deterministic symbolic safeguards—multi-tier chemical and pharmacological policy gates, syntactic telemetry dependency parsing with bijective unit invariance, aqueous ammonia equilibrium calculations, and directional NLI verification—with edge-oriented neural models (`BAAI/bge-m3`, fine-tuned `DeBERTa-v3-small`, quantized `Qwen-2.5-1.5B-Instruct` GGUF, offline `NLLB-200`, and local `Whisper`).

---

## System Architecture

AquaAssist executes a **Dual-Path Neuro-Symbolic Pipeline** (`app.py`: `run_aquaassist_pipeline`) that separates sub-20 ms deterministic policy/telemetry checks from evidence-grounded retrieval, NLI verification, and constrained response synthesis, routing every inquiry into `ANSWER`, `CLARIFY`, or `ABSTAIN`:

```text
               [Farmer Input: Typed English / Telugu / Tenglish or Audio]
                                           │
                                           ▼
                     [Step 1: FFmpeg 16 kHz Mono + Whisper (base)]
                      (Translates spoken Telugu/English to English)
                                           │
                                           ▼
                   [Step 2: Conversational Coreference & State Guard]
                   (Resolves elliptical turns; shields raw telemetry)
                                           │
                                           ▼
                   [Step 3: Tier 1–4 Policy & Chemical Safety Router]
                   • Tier 1/1b: Domain Membership & Absurdity Guard
                   • Tier 2: Banned Drugs, Antibiotic Rx, Folk Cures,
                             Biocide/Blind Dosing & Chemical Mixing
                   • Tier 3: Live Market Prices, Brands, Loans/Subsidy
                   • Tier 4: Vague Complaints & Missing Context
                                           │
                         ┌─────────────────┴─────────────────┐
                         │ (Policy Intercept)                │ (Pass)
                         ▼                                   ▼
            [Deterministic Bilingual             [Step 4: Intent & Telemetry]
             Policy Advisory Output]             (spaCy Dependency Tree, Noise
             (ABSTAIN / CLARIFY)                  Filter & Bijective Units)
                         │                                   │
                         │                                   ▼
                         │                       [Step 5: Telemetry & Reflexes]
                         │                       • Concurrent Sensor Conflict
                         │                         vs. Diurnal Epoch Cycle
                         │                       • Physical Plausibility Bounds
                         │                       • Acute Hypoxia (< 3.0 mg/L)
                         │                       • Marginal DO (3.0–5.0 mg/L)
                         │                       • Emerson Toxic NH₃ (>= 0.05)
                         │                       • Optimal DO Fast-Path (>= 5.0)
                         │                                   │
                         │                 ┌─────────────────┴─────────────────┐
                         │                 │ (Fast-Path Trigger)               │ (Slow-Path RAG)
                         │                 ▼                                   ▼
                         │    [Bilingual Telemetry               [Step 6: ChromaDB Retrieval]
                         │     Emergency / Status]               (BAAI/bge-m3 • 1,971 Chunks •
                         │                                        Species & Topic Isolation)
                         │                 │                                   │
                         │                 │                                   ▼
                         │                 │                     [Step 7: DeBERTa-v3 NLI Gate]
                         │                 │                     (Declarative Hypothesis Test:
                         │                 │                      ANSWER / CLARIFY / ABSTAIN)
                         │                 │                                   │
                         │                 │                                   ▼
                         │                 │                     [Step 8: Qwen 2.5 1.5B GGUF]
                         │                 │                     (Token-Budgeted Context +
                         │                 │                      Clinical Directives + Fallback)
                         │                 │                                   │
                         │                 │                                   ▼
                         │                 │                     [Offline NLLB-200 Translator]
                         │                 │                     (Unit-Shielded Greedy Telugu +
                         │                 │                      Aquaculture Calque Cleaner)
                         └─────────────────┴─────────────────┬─────────────────┘
                                                             ▼
                                       [Bilingual Advisory + Edge-TTS + Audit Log]
                                        (te-IN-ShrutiNeural / en-IN-NeerjaNeural)
```

---

## What Happens Inside the Pipeline

### 1. Multimodal Input & Multi-Turn Session Management (`app.py`, `src/whisper_transcriber.py`)
- **Voice & Multilingual Text Input:** Standardizes microphone/uploaded audio to 16 kHz mono 16-bit PCM WAV via `ffmpeg` and transcribes/translates speech directly to English via local OpenAI `Whisper` (`base`, `task="translate"`). Typed English, Telugu Unicode (`తెలుగు`), and Romanized Telugu-English (`Tenglish`, e.g., *"royyalu baagolevu"*, *"cheruvulo DO taggindi"*) are accepted directly without UI blocking.
- **Coreference Resolution with Telemetry Isolation:** Resolves short elliptical follow-ups (*"what should I do about that now?"*, *"how to fix it?"*) using per-session diagnostic state (`species`, `habitat`, `telemetry`, `symptoms`, `pending_metrics`), while bypassing expansion whenever numeric readings are reported so historical values never contaminate current-turn telemetry.
- **Spoken Output & Consultation Logs:** Synthesizes MD5-cached audio via `edge-tts` (`te-IN-ShrutiNeural` for Telugu advisories, `en-IN-NeerjaNeural` fallback), exports downloadable consultation reports (`AquaAssist_Diagnostic_Report.txt`), and exposes live retrieval distances, NLI probabilities, and active state in an inspectable UI accordion.

### 2. Four-Tier Deterministic Policy & Chemical Safety Router (`src/decision_router.py`, `src/decision_response.py`)
Evaluates raw user queries in `< 20 ms` prior to vector search or SLM generation:
- **Tier 1 & 1b — Aquaculture Domain & Biological Absurdity Filter:** Validates aquaculture membership across lemmatized English (`spaCy` `en_core_web_sm`), Telugu script, Romanized Telugu inflections, telemetry units, and state-aware slot-filling numbers; rejects terrestrial crop agronomy (`wheat`, `paddy`, `cotton`, `tractor`, `వరి పంట`) and biological absurdities (*"why are shrimps not flying?"*).
- **Tier 2 — Pharmacological, Chemical & Corpus-Scope Guardrails:**
  - **Banned Substances (`BANNED_SUBSTANCE_REFUSAL`):** Blocks prohibited veterinary drugs and pesticides (`chloramphenicol`, `nitrofurans`, `furazolidone`, `malachite green`, `endosulfan`, `dichlorvos`/`Nuvan`, `chlorpyrifos`, `cypermethrin`, `sodium metabisulfite`, `పురుగుమందు`, `నువాన్`).
  - **Antibiotic & Therapeutic Biocide Refusals (`ANTIBIOTIC_PRESCRIPTION_REFUSAL`, `THERAPEUTIC_DOSE_REFUSAL`):** Refuses antibiotic/oxytetracycline prescriptions and uncalibrated therapeutic dosing of `formalin`, `copper sulfate`, or `potassium permanganate` (`KMnO₄`).
  - **Hazardous Chemical Mixing & Stocked-Pond Disinfection (`CHEMICAL_HAZARD`):** Prevents incompatible oxidizing/reducing combinations (e.g., bleaching powder + `KMnO₄`, formalin, ammonia, or probiotics; `CuSO₄` + formalin) and direct bleaching powder disinfection in stocked ponds.
  - **Blind Conditioner Dosing (`BLIND_DOSING_REFUSAL`):** Withholds lime (`sunnam`), zeolite, dolomite, or mineral dosages when pond area/depth/volume is missing or active mortality is reported.
  - **Unproven Remedies, Evasion & Out-of-Scope Systems:** Intercepts unverified folk cures (turmeric/`pasupu`, neem leaves, common salt for EHP), residue-test evasion or concealing diseased stock, yield guarantees, and out-of-corpus systems (ornamental goldfish, scampi, cage culture, freshwater Rohu/Catla polyculture).
- **Tier 3 — Temporal & Commercial Boundary (`REALTIME_DATA_LIMITATION`):** Intercepts live village/mandi shrimp pricing, feed/hatchery brand rankings, bank loans, subsidies, and insurance queries.
- **Tier 4 — Vague Diagnostic & Missing Context Gate (`VAGUE_DIAGNOSTIC`):** Requests water appearance, stock behavior, and test-kit readings when complaints lack specific symptoms/measurements, report naked unit-only numbers (*"the reading is 4.5 mg/L"*), or ask about TAN toxicity without `pH`.

### 3. Syntactic Telemetry Extraction & Limnological Reflexes (`src/intent_extractor.py`, `src/diagnostic_engine.py`)
- **Dependency Parsing & Bijective Unit Invariance:** Extracts 11 canonical parameters (`DO`, `pH`, `TAN`, `NO2`, `NO3`, `alkalinity`, `temperature`, `salinity`, `transparency`, `depth`, `ORP`) across copular, prepositional, and proximity structures while decoupling hardware nouns (`probe`, `meter`, `sensor`, `kit`). Enforces bijective physical unit mappings (`ppt`/`psu` → `salinity`, `°C` → `temperature`) and prevents concentration units (`mg/L`, `ppm`) from misattaching to `pH` or `temperature`.
- **Clock & Farm-Metadata Noise Filter:** Excludes timestamps (`5:30`, `4:45am`, `gantalaki`, `గంటలకు`) and farm operational figures (`DOC 78`, `pond 4`, `4 trays`, `3 acre`, `count 40`, `1.5 hr`) from overwriting water-quality slots.
- **Diurnal Cycle vs. Sensor Conflict Detection:** Indexes temporal markers (`morning`, `afternoon`, `dawn`, `night`, `am`, `pm`) per reading. Concurrent diverging measurements (`DO` spread `>= 2.0 mg/L` or `pH` spread `>= 1.5` in the same epoch) trigger `CONFLICT_READINGS` sensor recalibration, whereas morning-to-afternoon swings are preserved as diurnal photosynthesis/respiration cycles.
- **Physical Bounds & Emergency/Status Reflexes:**
  - Rejects physically implausible sensor values (`DO`: `0.0–22.0 mg/L`, `pH`: `3.5–11.5`, `temperature`: `5.0–45.0 °C`, `salinity`: `0.0–60.0 ppt`, `TAN`: `0.0–20.0 mg/L`, `alkalinity`: `10.0–600.0 mg/L`) via `SENSOR_CONFLICT`.
  - **Acute Hypoxia (`DO < 3.0 mg/L`):** Triggers an immediate bilingual aeration and feed-stoppage alarm on fresh-turn readings without invoking retrieval or SLM generation, while `MARGINAL_DO_CLARIFY` flags unexplained `3.0–5.0 mg/L` readings in the routing engine.
  - **Emerson Un-ionized Ammonia (`NH₃`) Equilibrium:** Computes aqueous $\mathrm{p}K_a = 0.09018 + \frac{2729.92}{273.15 + T}$ and toxic un-ionized $\mathrm{NH}_3$ from `TAN`, `pH`, and temperature (default $28\,^\circ\mathrm{C}$), flagging $\mathrm{NH}_3 \ge 0.05\,\mathrm{mg/L}$ (`EMERGENCY_AMMONIA` in the diagnostic engine and `[CRITICAL CHEMICAL INVARIANT]` context injection during response synthesis, including warnings against applying lime at high `pH`).
  - **Optimal DO Fast-Path (`DO >= 5.0 mg/L`):** Confirms healthy oxygenation immediately when no clinical symptoms are reported.

### 4. Knowledge Ingestion, Fine-Tuned NLI Gate & Grounded Generation (`src/ingest.py`, `src/diagnostic_engine.py`, `src/response_generator.py`)
- **Faceted ChromaDB Knowledge Store:** **1,971 semantic chunks** indexed in `data/chroma_db/` (`aqua_assist` collection) from **19 contributing manuals across 23 ICAR-CIBA, FAO, and aquaculture reference PDFs** (`data/raw_pdfs/`). Offline ingestion (`src/ingest.py`) extracts `spaCy` Subject-Predicate-Object triplets, applies `constraints.gbnf` JSON grammar with `Qwen-2.5-1.5B` for structured metadata extraction, and embeds chunks with `BAAI/bge-m3` partitioned by `species` (`shrimp`, `fish`, `general`), `habitat` (`brackish`, `freshwater`, `general`), and `primary_topic` (with hatchery `artemia`/`nauplii` exclusion on grow-out pond checks).
- **Fine-Tuned DeBERTa-v3 NLI Verification (`src/generate_rlaif_data.py`, `src/train_router.py`):** Rewrites interrogative queries into declarative propositions and scores `[chunk_text, declarative_claim]` pairs using `cross-encoder/nli-deberta-v3-small` fine-tuned on **5,349 balanced three-way RLAIF triplets** (`data/evaluation/rlaif_dataset.json`: 1,783 `Entailment`/`ANSWER`, 1,783 `Neutral`/`CLARIFY`, 1,783 `Contradiction`/`ABSTAIN`). `make_decision()` abstains on contradicted claims (`contradiction >= 0.35` on top chunk or `>= 0.40` across 2 of top 3) and requires positive evidential entailment (`>= 0.65` primary or `>= 0.25` with `distance < 0.75` and `contradiction < 0.25`) to `ANSWER`.
- **Token-Budgeted SLM Synthesis, Extractive Fallback & Offline Translation:** Formats up to 3 verified chunks (`<= 150` words/chunk) for `Qwen-2.5-1.5B-Instruct` (`llama-cpp-python`, `n_ctx=2048`, `n_threads=2`, Metal GPU offload on Apple Silicon) with clinical prompt directives (WSSV non-antibiotic viral biosecurity, *P. vannamei* `15–25 ppt` optimal vs. `2–40 ppt` survival tolerance, diurnal `pH` agricultural gypsum guidance, polarity contradiction sanitization) and a deterministic extractive prose fallback if GGUF weights are absent. Translates offline via `facebook/nllb-200-distilled-600M` (`local_files_only=True`, `num_beams=1`, scientific unit shielding for `ppt`/`mg/L`/`ppm`, and Telugu aquaculture calque post-processing).
- **Atomic Blue/Green OTA Sync (`src/sync_manager.py`):** Bootstraps missing `chroma_db.zip` and `deberta_aquaculture_router.zip` release bundles on first startup and swaps versioned directories via an atomic `data/active_db.json` pointer to avoid file-locking errors.

---

## Evaluation & Stress-Test Benchmarks

AquaAssist has been evaluated across static architectural audits, deterministic/slow-path scenario regressions, a structured 258-query multilingual benchmark, and an unscripted 260-query noisy field stress test:

| Evaluation Suite | Runner | Scope & Languages | Result |
| :--- | :--- | :--- | :--- |
| **Architecture 2 Static Audit** | `testing/verify_architecture.py` | AST & pattern audit across `src/` and `app.py` (no hardcoded answers, real NLI, state & OTA invariants) | **PASSED (0 violations)** |
| **20-Scenario Clinical & Latency Suite** | `testing/test_rigorous_eval_full.py` | `T1`–`T20` (`25OCT_after-stresstest-correction_results/benchmark_results_T1_T20.json`) | **20 / 20 (100.0%)** (`4.48–17.61 ms` fast-path; `4.05–15.38 s` slow-path) |
| **Structured 258-Query Benchmark (Calibrated Router + NLI)** | `testing/run_blind_evaluation.py` | `AQ-001`–`AQ-258` across 30 categories (142 English, 64 Romanized Te-En, 52 Telugu) | **248 / 258 (96.12%)** ($\kappa = 0.941$, macro-F1 $= 0.958$, `189.1 ms/query`, `86/86` `ABSTAIN` recall) |
| **Structured 258-Query Benchmark (End-to-End UI Pipeline)** | `batch_evaluate_aquaassist.py` | Full `app.run_aquaassist_pipeline` (`Qwen-2.5` + `NLLB-200`) on `AQ-001`–`AQ-258` | **210 / 258 (81.40%)** ($\kappa = 0.711$, macro-F1 $= 0.802$, `0` blank outputs; 38-query gap due to UI banner string-heuristic mapping & `app.py` Step 5 flag wiring) |
| **Unscripted 260-Query Real-Life Messy Stress Test** | `batch_evaluate_aquaassist.py` | `MQ-001`–`MQ-260` noisy SMS English (101), colloquial `Tenglish` (91), & Telugu (68) | **148 / 260 (56.92%)** post-hardening (up from `133 / 260`, `51.15%` pre-fix; $\kappa = 0.308$, `0` blank outputs) |

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
│   ├── decision_response.py            # Bilingual (English & Telugu) policy advisory templates
│   ├── diagnostic_engine.py            # ChromaDB faceted retrieval, Emerson NH₃ math, & DeBERTa NLI gate
│   ├── response_generator.py           # Token-budgeted Qwen 2.5 GGUF synthesis & offline NLLB translation
│   ├── whisper_transcriber.py          # FFmpeg 16 kHz resampler & OpenAI Whisper speech-to-English
│   ├── sync_manager.py                 # Atomic blue/green pointer-based OTA updater (ChromaDB & DeBERTa)
│   ├── ingest.py                       # PDF manual parser, triplet extractor, GBNF synthesizer, & indexer
│   ├── embed.py                        # Standalone BAAI/bge-m3 chunk embedder
│   ├── generate_rlaif_data.py          # 3-way RLAIF NLI triplet generator (5,349 pairs)
│   ├── train_router.py                 # Fine-tuning script for DeBERTa-v3-small aquaculture Cross-Encoder
│   ├── cross_encoder_gate.py           # Standalone Cross-Encoder retrieval verification utility
│   ├── persona_renderer.py             # CLI bilingual field advisory formatter
│   ├── run_ablation.py                 # Baseline NLI routing ablation & confusion matrix evaluator
│   ├── measure_latency.py              # BGE-M3 + ChromaDB retrieval latency profiler
│   ├── profile_qwen_latency.py         # Qwen GGUF memory & token latency profiler
│   ├── plot_latency.py                 # Retrieval latency figure generator
│   └── plot_safety_coverage.py         # Safety invariant coverage figure generator
├── testing/
│   ├── verify_architecture.py          # Static Architecture 2 compliance auditor
│   ├── test_rigorous_eval_full.py      # Formal 20-scenario (T1–T20) clinical, multi-turn & latency suite
│   ├── run_blind_evaluation.py         # Calibrated 258-query blind routing & NLI evaluator
│   ├── test_rc01_telemetry.py          # Telemetry unit invariance & timestamp/metadata exclusion tests
│   ├── test_rc02_schema_ammonia.py     # ChromaDB topic schema & Emerson toxic ammonia tests
│   ├── test_rc03_rc04_router.py        # Tenglish stems, banned substances, & vague gate tests
│   ├── test_rc05_rc07_generator.py     # Extractive fallback, polarity sanitizer, & Telugu calque tests
│   ├── test_aquaculture_invariants.py  # Biological boundary invariant unit tests
│   ├── test_intent_extractor.py        # Intent & conflicting measurement unit tests
│   ├── test_retrieval.py               # Direct ChromaDB retrieval sanity check
│   └── test_task1.py .. test_task4.py  # Component milestone verification tests
├── data/
│   ├── raw_pdfs/                       # 23 ICAR-CIBA, FAO, & aquaculture reference PDFs (19 indexed)
│   ├── processed/chunks.json           # Processed text chunks
│   ├── chroma_db/                      # Pre-indexed ChromaDB store (aqua_assist, 1,971 chunks)
│   ├── evaluation/                     # RLAIF dataset (5,349 pairs) & latency/memory profiling JSONs
│   └── version.json                    # Knowledge base release metadata
└── models/                             # Local weights (fetched via OTA or provisioned locally)
    ├── deberta_aquaculture_router/     # Fine-tuned 3-class DeBERTa-v3-small Cross-Encoder checkpoint
    └── qwen-2.5-1.5b-instruct.gguf     # Quantized Qwen 2.5 1.5B Instruct GGUF weights
```
*(Local evaluation workbooks in `messy_query_excelsheets/` and `real_life_messy_queries/`, along with execution traces in `24OCT_real_life_query_test_results/` and `25OCT_after-stresstest-correction_results/`, are excluded from Git via `.gitignore`.)*

---

## Getting Started

### 1. Prerequisites
- **Python 3.10+** (macOS Apple Silicon MPS/Metal, Linux CUDA, or CPU; thread-bounded to 2 threads for edge/Spaces CPU tiers)
- **`ffmpeg`** (required for microphone/uploaded audio resampling)

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

### 3. Models & Knowledge Base Provisioning
- **Automatic OTA Bootstrap:** On first launch of `app.py`, `src/sync_manager.py` automatically downloads and extracts `chroma_db.zip` and `deberta_aquaculture_router.zip` from GitHub Releases if not present locally.
- **Local SLM Weights:** Place `qwen-2.5-1.5b-instruct.gguf` inside `models/` for dynamic generative synthesis (if absent, `src/response_generator.py` degrades cleanly to grounded extractive prose fallback).
- **Offline Model Cache:** Cache `facebook/nllb-200-distilled-600M` and `BAAI/bge-m3` once in `~/.cache/huggingface` while online prior to offline field operation (`src/response_generator.py` enforces `local_files_only=True` at runtime).
- **Re-ingesting Raw Manuals / Re-training Router (Optional):**
  ```bash
  python src/ingest.py
  python src/generate_rlaif_data.py
  python src/train_router.py
  ```

### 4. Running the Application

#### Option A: Direct Python (Gradio UI)
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

#### Option B: Docker Compose + Cloudflare HTTPS Tunnel
```bash
docker compose up --build -d

# Inspect the public trycloudflare.com HTTPS URL:
docker logs aquaassist-tunnel
```

---

## Running Verification & Benchmarks

```bash
# 1. Static Architecture 2 compliance audit
python testing/verify_architecture.py

# 2. Formal 20-scenario clinical, multi-turn & latency benchmark (T1–T20)
python testing/test_rigorous_eval_full.py

# 3. Calibrated 258-query blind routing & NLI benchmark (AQ-001 to AQ-258)
python testing/run_blind_evaluation.py

# 4. End-to-end generative pipeline evaluation on Excel workbooks (writes *_Executed.xlsx & .jsonl traces)
python batch_evaluate_aquaassist.py --folder messy_query_excelsheets
python batch_evaluate_aquaassist.py --folder real_life_messy_queries
```
