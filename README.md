# Evidence-Aware Vernacular RAG (EAV-RAG)

## Abstract
AI adoption in high-stakes agricultural domains, such as aquaculture, is constrained by unreliable internet connectivity and the vulnerability of edge-deployed Small Language Models (SLMs) to hallucination. This repository contains the source code for an Evidence-Aware Vernacular RAG (EAV-RAG) architecture. Designed to operate 100% offline on standard edge hardware, the system mitigates cross-lingual hallucinations (Telugu-English) by replacing standard generative guessing with a mathematically rigorous Natural Language Inference (NLI) Cross-Encoder gate, trained via an automated Reinforcement Learning from AI Feedback (RLAIF) data engine.

## Repository Architecture
* `src/`: Core Python executable scripts for ingestion, embedding, and model orchestration.
* `data/`: Directory for data pipelines. 
  * *Note: Raw PDFs and ChromaDB vector indices are locally isolated and ignored by version control to preserve repository integrity. Processed JSON chunks are tracked.*
* `docs/`: Academic documentation, technical flowcharts, and IEEE manuscript drafts.

## Initial Setup & Installation
Contributors must execute the following protocol to synchronize their local environments:

1. **Clone the Repository**
   git clone https://github.com/Khushhiii08/AquaAssist.git
   cd AquaAssist

2. **Initialize Virtual Environment**
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

3. **Install Dependencies**
   pip install -r requirements.txt

## Execution Pipeline: Phase 1 (Corpus Curation)
The initial phase focuses on the ingestion and semantic chunking of official ICAR-CIBA aquaculture manuals. 
* `src/ingest.py`: Parses raw PDFs, executes overlapping semantic chunking, and appends rule-based metadata.
* `src/embed.py`: Utilizes the BGE-M3 multilingual model to convert text chunks into dense vectors, storing them in a local ChromaDB instance.

## Project Contributors
* **Khushi** - Core ML Architecture & Pipeline Implementation
* **Meghana** - Integration & Retrieval Testing
* **Jyothish** - Project Management, Documentation & Visual Architecture
* **Praisy** - Data Preprocessing & Pipeline Execution