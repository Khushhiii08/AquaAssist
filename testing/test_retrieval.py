import sys
sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------
# SETTINGS
# -----------------------------
DB_PATH = "data/chroma_db"
COLLECTION_NAME = "aquaculture_knowledge"
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5

# -----------------------------
# VERIFIED TEST SET
# -----------------------------
TEST_QUERIES = [
    {
        "id": "Q01",
        "query": "In the sewage-fed aquaculture pond study, how did dissolved oxygen change from the sewage inlet (L1) to the farthest location (L3)?",
        "expected_source": "augmenting_aquaculture_production.pdf",
        "expected_chunks": ["CHK-00016", "CHK-00017"],
    },
    {
        "id": "Q02",
        "query": "What pH ranges were reported at L1, L2 and L3 in the selected sewage-fed aquaculture ponds?",
        "expected_source": "augmenting_aquaculture_production.pdf",
        "expected_chunks": ["CHK-00016", "CHK-00017"],
    },
    {
        "id": "Q03",
        "query": "What monthly mean water-temperature range was observed in the selected sewage-fed ponds, and in which months were the lowest and highest means reported?",
        "expected_source": "augmenting_aquaculture_production.pdf",
        "expected_chunks": ["CHK-00014"],
    },
    {
        "id": "Q04",
        "query": "How did total ammoniacal nitrogen (TAN) change from L1 to L3 in the sewage-fed pond study?",
        "expected_source": "augmenting_aquaculture_production.pdf",
        "expected_chunks": ["CHK-00017", "CHK-00018"],
    },
    {
        "id": "Q05",
        "query": "How did nitrite nitrogen change from L1 to L3 in the same sewage-fed pond study?",
        "expected_source": "augmenting_aquaculture_production.pdf",
        "expected_chunks": ["CHK-00017"],
    },
    {
        "id": "Q06",
        "query": "What total alkalinity values were reported at L1, L2 and L3 in the selected ponds?",
        "expected_source": "augmenting_aquaculture_production.pdf",
        "expected_chunks": ["CHK-00016", "CHK-00017"],
    },
    {
        "id": "Q07",
        "query": "How was water-hyacinth coverage managed after stocking in the sewage-fed fish ponds?",
        "expected_source": "augmenting_aquaculture_production.pdf",
        "expected_chunks": ["CHK-00007", "CHK-00008"],
    },
    {
        "id": "Q08",
        "query": "When no aeration devices were provided in the sewage-fed ponds, what operation was used to improve dissolved oxygen?",
        "expected_source": "augmenting_aquaculture_production.pdf",
        "expected_chunks": ["CHK-00013"],
    },
    {
        "id": "Q09",
        "query": "Which diseases were Indian white shrimp brooders screened for before being shifted to the hatchery?",
        "expected_source": "fish_breeding_book.pdf",
        "expected_chunks": ["CHK-00438"],
    },
    {
        "id": "Q10",
        "query": "How is biosecurity defined in the FAO aquatic-animal-health guidance contained in the corpus?",
        "expected_source": "fao_technical_guidelines_for_responsible_fisheries.pdf",
        "expected_chunks": ["CHK-00245"],
    },
    {
        "id": "Q11",
        "query": "What role did shrimp farmer groups play in disease prevention according to the shrimp farmer-group study?",
        "expected_source": "extension_methodology_for_assessing.pdf",
        "expected_chunks": ["CHK-00186"],
    },
    {
        "id": "Q12",
        "query": "What analytical approach was used to estimate technical efficiency of shrimp farms in Andhra Pradesh?",
        "expected_source": "technical_efficiency_analysis.pdf",
        "expected_chunks": ["CHK-00490"],
    },
    {
        "id": "Q13",
        "query": "Which water-quality parameters are explicitly named in the study description of the sewage-fed aquaculture ponds?",
        "expected_source": "augmenting_aquaculture_production.pdf",
        "expected_chunks": ["CHK-00008"],
    },
    {
        "id": "Q14",
        "query": "Was supplementary feeding and fertilization used in the sewage-fed fish ponds described in the corpus?",
        "expected_source": "augmenting_aquaculture_production.pdf",
        "expected_chunks": ["CHK-00013"],
    },
    {
        "id": "Q15",
        "query": "What shrimp-farming disease event in Andhra Pradesh is described as causing repeated crop losses and leaving much developed area fallow?",
        "expected_source": "brackish_water_aquaculture.pdf",
        "expected_chunks": ["CHK-00034"],
    },
]

# -----------------------------
# LOAD MODEL + DATABASE
# -----------------------------
print("Loading BAAI/bge-m3...")
model = SentenceTransformer(MODEL_NAME)

print("Opening ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

print("Collection:", collection.name)
print("Stored chunks:", collection.count())
print("=" * 80)

# -----------------------------
# RETRIEVAL TEST
# -----------------------------
all_results = []

for item in TEST_QUERIES:

    print("\n" + "=" * 80)
    print(item["id"])
    print("QUERY:", item["query"])
    print("=" * 80)

    embedding = model.encode(
        [item["query"]],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_ids = results["ids"][0]

    # Check whether expected chunk appears in Top-5
    matched_ranks = []

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in item["expected_chunks"]:
            matched_ranks.append(rank)

    if matched_ranks:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    print("VERDICT:", verdict)
    print("Expected chunks:", item["expected_chunks"])
    print("Matched ranks:", matched_ranks)

    # Print Top-5
    for rank in range(TOP_K):

        chunk_id = results["ids"][0][rank]
        source = results["metadatas"][0][rank].get(
            "source", "N/A"
        )
        distance = results["distances"][0][rank]
        text = results["documents"][0][rank]

        print(f"\n--- Rank {rank + 1} ---")
        print("Chunk ID:", chunk_id)
        print("Source:", source)
        print("Distance:", distance)
        print("Text:", text)

    # Store result for later Excel generation
    all_results.append({
        "id": item["id"],
        "query": item["query"],
        "expected_source": item["expected_source"],
        "expected_chunks": item["expected_chunks"],
        "retrieved_ids": retrieved_ids,
        "matched_ranks": matched_ranks,
        "verdict": verdict
    })

# -----------------------------
# SUMMARY
# -----------------------------
print("\n\n" + "=" * 80)
print("FINAL RETRIEVAL SUMMARY")
print("=" * 80)

for result in all_results:
    print(
        result["id"],
        "->",
        result["verdict"],
        "| Retrieved:",
        result["retrieved_ids"]
    )

print("\nTesting completed successfully.")