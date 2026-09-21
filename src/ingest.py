import os
import json
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"

import chromadb
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(4)

RAW_PDFS_DIR = "./data/raw_pdfs"
PROCESSED_DATA_DIR = "./data/processed"
CHROMA_DB_DIR = "./data/chroma_db"
CHUNKS_PATH = os.path.join(PROCESSED_DATA_DIR, "chunks.json")
MIN_WORDS = 8      # Blocks single-line headers and tiny orphan fragments
MIN_CHARS = 40     # Blocks short punctuation-heavy artifacts
MAX_WORDS = 50


def word_count(text):
    return len(re.findall(r"\S+", text))

def split_into_bounded_chunks(text, max_words=MAX_WORDS):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_words = []

    for sentence in sentences:
        sentence_words = sentence.split()

        if not sentence_words:
            continue

        if len(sentence_words) > max_words:
            if current_words:
                chunk_text = " ".join(current_words)
                # Apply quality filter before appending
                if len(chunk_text.split()) >= MIN_WORDS and len(chunk_text) >= MIN_CHARS:
                    chunks.append(chunk_text)
                current_words = []

            for start in range(0, len(sentence_words), max_words):
                sub_chunk = " ".join(sentence_words[start:start + max_words])
                if len(sub_chunk.split()) >= MIN_WORDS and len(sub_chunk) >= MIN_CHARS:
                    chunks.append(sub_chunk)

            continue

        if len(current_words) + len(sentence_words) <= max_words:
            current_words.extend(sentence_words)
        else:
            if current_words:
                chunk_text = " ".join(current_words)
                if len(chunk_text.split()) >= MIN_WORDS and len(chunk_text) >= MIN_CHARS:
                    chunks.append(chunk_text)

            current_words = sentence_words.copy()

    if current_words:
        chunk_text = " ".join(current_words)
        if len(chunk_text.split()) >= MIN_WORDS and len(chunk_text) >= MIN_CHARS:
            chunks.append(chunk_text)

    return chunks

def classify_category(text):
    lower_text = text.lower()

    if any(
        word in lower_text
        for word in [
            "wssv",
            "ehp",
            "vibrio",
            "disease",
            "pathogen",
            "mortality",
            "syndrome",
        ]
    ):
        return "Disease Diagnostics"

    if any(
        word in lower_text
        for word in [
            "oxygen",
            "ph",
            "salinity",
            "ammonia",
            "tan",
            "alkalinity",
            "temperature",
            "ppm",
            "ppt",
        ]
    ):
        return "Water Quality Limits"

    if any(
        word in lower_text
        for word in [
            "dosage",
            "treatment",
            "probiotic",
            "application",
            "gram",
            "ml",
            "disinfection",
        ]
    ):
        return "Dosage & Treatment"

    return "Husbandry & Management"


def detect_species(text):
    lower_text = text.lower()

    if "vannamei" in lower_text:
        return "Litopenaeus vannamei"

    if "rohu" in lower_text or "labeo rohita" in lower_text:
        return "Labeo rohita"

    return "General Aquaculture"


def extract_parameters(text):
    lower_text = text.lower()
    parameters = {}

    ph_match = re.search(
        r"\bph\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
        lower_text,
    )

    if ph_match:
        parameters["pH"] = ph_match.group(1)

    do_match = re.search(
        r"(?:dissolved\s+oxygen|\bdo\b)"
        r"\s*(?:=|:|of|at)?\s*"
        r"([0-9]+(?:\.[0-9]+)?)"
        r"\s*(?:mg/l|mg/litre|ppm)?",
        lower_text,
    )

    if do_match:
        parameters["Dissolved_Oxygen"] = do_match.group(1)

    return parameters


def extract_metadata_and_chunk(text, source_filename):
    bounded_chunks = split_into_bounded_chunks(text)

    chunks = []

    for chunk_text in bounded_chunks:
        chunks.append(
            {
                "source_document": source_filename,
                "domain_category": classify_category(chunk_text),
                "target_species": detect_species(chunk_text),
                "extracted_parameters": extract_parameters(chunk_text),
                "text": chunk_text,
            }
        )

    return chunks


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def run_hybrid_ingestion():
    print("[*] Initializing corpus ingestion and indexing...")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    all_chunks = []

    if not os.path.exists(RAW_PDFS_DIR) or not os.listdir(RAW_PDFS_DIR):
        print(f"[!] Warning: No source files found in {RAW_PDFS_DIR}.")

        os.makedirs(RAW_PDFS_DIR, exist_ok=True)

        sample_path = os.path.join(
            RAW_PDFS_DIR,
            "ICAR-CIBA_Sample_Guidelines.txt",
        )

        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(
                "ICAR-CIBA guidelines for Litopenaeus vannamei culture. "
                "Optimal dissolved oxygen must be maintained above 4.0 mg/L. "
                "pH range should be 7.5 to 8.5. "
                "For White Spot Syndrome Virus (WSSV), immediate quarantine "
                "and strict biosecurity protocols are required. "
                "Total Ammonia Nitrogen (TAN) must not exceed 0.05 mg/L."
            )

    global_chunk_counter = 0

    for filename in sorted(os.listdir(RAW_PDFS_DIR)):
        file_path = os.path.join(RAW_PDFS_DIR, filename)

        if not os.path.isfile(file_path):
            continue

        lower_filename = filename.lower()
        content = ""

        if lower_filename.endswith(".txt") or lower_filename.endswith(".md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"[-] Error reading {filename}: {e}")
                continue

        elif lower_filename.endswith(".pdf"):
            try:
                import pypdf

                reader = pypdf.PdfReader(file_path)
                page_texts = []

                for page in reader.pages:
                    page_texts.append(page.extract_text() or "")

                content = "\n\n".join(page_texts)

            except Exception as e:
                print(f"[-] Error reading PDF {filename}: {e}")
                continue

        else:
            print(f"[*] Skipping unsupported file: {filename}")
            continue

        file_chunks = extract_metadata_and_chunk(
            content,
            filename,
        )

        print(
            f"[+] {filename}: "
            f"{len(file_chunks)} chunks generated"
        )

        for chunk in file_chunks:
            chunk["chunk_id"] = f"CHK-{global_chunk_counter:05d}"
            global_chunk_counter += 1
            all_chunks.append(chunk)

    if not all_chunks:
        raise ValueError("[-] No text chunks generated.")

    oversized = [
        chunk
        for chunk in all_chunks
        if word_count(chunk["text"]) > MAX_WORDS
    ]

    if oversized:
        raise ValueError(
            f"[-] Found {len(oversized)} chunks over "
            f"{MAX_WORDS} words."
        )

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[+] Saved {len(all_chunks)} chunks to "
        f"{CHUNKS_PATH}"
    )

    max_words_found = max(
        word_count(chunk["text"])
        for chunk in all_chunks
    )

    print(
        f"[+] Maximum chunk size: "
        f"{max_words_found} words"
    )

    device = get_optimal_device()

    print(
        f"[*] Loading BAAI/bge-m3 model. "
        f"Device: {device.upper()}"
    )

    embed_model = SentenceTransformer(
        "BAAI/bge-m3",
        device=device,
    )

    print(
        f"[*] Connecting to ChromaDB at "
        f"{CHROMA_DB_DIR}..."
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR
    )

    collection_name = "aquaculture_knowledge"

    try:
        client.delete_collection(collection_name)
        print("[*] Deleted old ChromaDB collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name
    )

    print("[*] Encoding chunks and inserting into ChromaDB...")

    batch_size = 256

    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]

        ids = [chunk["chunk_id"] for chunk in batch]
        texts = [chunk["text"] for chunk in batch]

        metadatas = [
            {
                "source": chunk["source_document"],
                "category": chunk["domain_category"],
                "species": chunk["target_species"],
                "parameters": json.dumps(
                    chunk["extracted_parameters"]
                ),
            }
            for chunk in batch
        ]

        embeddings = embed_model.encode(
            texts,
            batch_size=8,
            show_progress_bar=True,
            device=device,
        ).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(
            f"[+] Indexed "
            f"{min(start + batch_size, len(all_chunks))}"
            f"/{len(all_chunks)} chunks"
        )

    print("[+] Ingestion complete!")

    print(
        f"[+] ChromaDB collection "
        f"'{collection_name}' contains "
        f"{collection.count()} chunks."
    )


if __name__ == "__main__":
    run_hybrid_ingestion()
