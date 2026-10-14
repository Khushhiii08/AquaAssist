import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


DB_PATH = "data/chroma_db"
COLLECTION_NAME = "aquaculture_knowledge"
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5

OUTPUT_FILE = Path("data/evaluation/latency_results.json")

TEST_QUERIES = [
    "What pH ranges were reported at L1, L2 and L3?",
    "How did dissolved oxygen change from L1 to L3?",
    "What diseases were Indian white shrimp brooders screened for?",
    "How is biosecurity defined in the FAO guidance?",
    "What role did shrimp farmer groups play in disease prevention?",
]


def main():

    print("=" * 60)
    print("AquaAssist Latency Measurement")
    print("=" * 60)

    # Load model
    print("\n[*] Loading BAAI/bge-m3...")

    start = time.perf_counter()

    model = SentenceTransformer(MODEL_NAME)

    model_load_time = time.perf_counter() - start

    print(
        f"[*] Model loading time: "
        f"{model_load_time:.3f} seconds"
    )

    # Open ChromaDB
    print("\n[*] Opening ChromaDB...")

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"[*] Collection: {collection.name}"
    )

    print(
        f"[*] Stored chunks: {collection.count()}"
    )

    results = []

    print("\n[*] Measuring queries...")

    for number, query in enumerate(
        TEST_QUERIES,
        start=1
    ):

        print(f"\nQuery {number}: {query}")

        # Embedding timing
        start = time.perf_counter()

        embedding = model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        embedding_time = time.perf_counter() - start

        # Retrieval timing
        start = time.perf_counter()

        collection.query(
            query_embeddings=embedding,
            n_results=TOP_K,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieval_time = time.perf_counter() - start

        total_time = (
            embedding_time
            + retrieval_time
        )

        results.append({
            "query_number": number,
            "query": query,
            "embedding_seconds": embedding_time,
            "retrieval_seconds": retrieval_time,
            "total_seconds": total_time
        })

        print(
            f"  Embedding: {embedding_time:.4f} s"
        )

        print(
            f"  Retrieval: {retrieval_time:.4f} s"
        )

        print(
            f"  Total:     {total_time:.4f} s"
        )

    # Calculate averages
    embedding_mean = sum(
        x["embedding_seconds"]
        for x in results
    ) / len(results)

    retrieval_mean = sum(
        x["retrieval_seconds"]
        for x in results
    ) / len(results)

    total_mean = sum(
        x["total_seconds"]
        for x in results
    ) / len(results)

    # Save results
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "number_of_queries": len(results),
        "model_load_seconds": model_load_time,
        "embedding_mean_seconds": embedding_mean,
        "retrieval_mean_seconds": retrieval_mean,
        "total_mean_seconds": total_mean,
        "queries": results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Final summary
    print("\n" + "=" * 60)
    print("LATENCY SUMMARY")
    print("=" * 60)

    print(
        f"Model loading: "
        f"{model_load_time:.4f} s"
    )

    print(
        f"Embedding mean: "
        f"{embedding_mean:.4f} s"
    )

    print(
        f"Retrieval mean: "
        f"{retrieval_mean:.4f} s"
    )

    print(
        f"Total mean: "
        f"{total_mean:.4f} s"
    )

    print("\n[*] Results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()

