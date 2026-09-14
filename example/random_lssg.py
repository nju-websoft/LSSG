"""Build and search an LSSG index with synthetic data.

Run from the repository root after installing the binding:

    python example/random_lssg.py
"""

import numpy as np

from pylssg import LSSGIndex


NUM_VECTORS = 100_000
DIMENSION = 128
NUM_LABELS = 3_862
MIN_LABELS_PER_VECTOR = 1
MAX_LABELS_PER_VECTOR = 5
NUM_QUERIES = 100
THREADS = 8
M = 16
EF_CONSTRUCTION = 128
SEARCH_EF = 100
K = 10


def make_random_label_sets(rng, count):
    sizes = rng.integers(MIN_LABELS_PER_VECTOR, MAX_LABELS_PER_VECTOR + 1, size=count)
    return [
        rng.choice(NUM_LABELS, size=int(size), replace=False).tolist()
        for size in sizes
    ]


def main():
    rng = np.random.default_rng(2026)
    vectors = rng.standard_normal((NUM_VECTORS, DIMENSION)).astype(np.float32)
    label_sets = make_random_label_sets(rng, NUM_VECTORS)

    # Reuse existing base label sets for the queries so every query has at
    # least one matching base vector under containment and equality.
    query_source_ids = rng.choice(NUM_VECTORS, size=NUM_QUERIES, replace=False)
    queries = rng.standard_normal((NUM_QUERIES, DIMENSION)).astype(np.float32)
    query_labels = [label_sets[int(index)] for index in query_source_ids]

    index = LSSGIndex(
        max_elements=NUM_VECTORS,
        dimension=DIMENSION,
        M=M,
        ef_construction=EF_CONSTRUCTION,
        space="l2",
    )

    print(f"Building {NUM_VECTORS:,} vectors with {THREADS} threads...")
    index.add_items(vectors, label_sets, threads=THREADS)
    print(f"Built index: {index.size:,} vectors, {index.dimension} dimensions")

    for filter_name in ("containment", "equality", "overlap"):
        ids, distances = index.search(
            queries,
            query_labels,
            k=K,
            ef=SEARCH_EF,
            filter=filter_name,
            threads=THREADS,
        )
        valid_results = np.count_nonzero(ids >= 0)
        print(
            f"{filter_name:11s}: ids={ids.shape}, distances={distances.shape}, "
            f"valid_results={valid_results}/{NUM_QUERIES * K}"
        )


if __name__ == "__main__":
    main()
