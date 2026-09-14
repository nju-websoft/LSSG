# Fast Label-Filtering Approximate Nearest Neighbor Search via Progressive Label Set Stratification (LSSG)

> This is the header-only library of LSSG, for the benchmarking repository, please check [here](https://github.com/ziqiwww/lssg_benchmark).

LSSG implements a multi-tier proximity graph for label-filtering approximate nearest neighbor search (LFANNS). It unifies the three canonical label filters — **equality**, **containment**, and **overlap** — in a single index by interpreting label-set relations through label-set distance (Jaccard distance). Edges are stratified into tiers with nested similarity bounds; the bottom tier is label-agnostic for global navigability, while upper tiers connect increasingly similar label sets and preserve equality exactly at the top tier. Search performs tier-ordered in-filtering beam search, escalating to stricter tiers to escape local minima in selective regions.

The accompanying executables and Bash files under `example/` are benchmark helpers.

Key contributions:

- A unified multi-tier graph supporting equality, containment, and overlap filters through progressive label-set stratification.
- Incremental insertion with dual-space pruning: RNG-style pruning in vector space and label diversification in label space.
- Two similar-label-set selection strategies: progressive **IVF scanning** (LSSG-IVF) and **MinHash LSH banding** (LSSG-MinHash) for scalability to large label spaces.
- Stepwise analysis of filter-valid expansion probabilities and conditional bounds on expected query cost.

## Requirements

1. C++20 compiler and CMake >= 3.20
2. OpenMP
3. Linux or another POSIX-like system with `mmap`/`madvise`
4. (Optional) AVX-512 for the SIMD-accelerated paths

## Installation

Just includes lssg/index.hh in your project. Build and search examples are also provided as below.

Build the project from a fresh build directory:

```bash
mkdir build && cd build
cmake .. && make -j
```

The default build is LSSG-MinHash. To build LSSG-IVF instead:

```bash
cmake .. -DLSSG_USE_MINHASH=OFF && make -j
```

The default build enables CPU-native SSE/AVX2/AVX-512 optimization flags when the compiler and host support them. Use `-DLSSG_ENABLE_NATIVE_SIMD=OFF` for a portable build. Native builds should be run on the same CPU feature class on which they were compiled.

The executables are created under `build/bin`:

- `build/bin/build_lssg`: builds and saves an LSSG index.
- `build/bin/search_lssg`: searches an LSSG index and reports recall/QPS.
- `build/bin/gen_label_gt`: internal helper invoked automatically when search ground truth is missing.

MinHash/IVF selection and the system-level defaults are centralized in [`lssg/config.hh`](lssg/config.hh). The default MinHash configuration is 64 hash functions, 16 bands, and 4 bits per hash. Index files record their format, variant, and MinHash configuration; rebuild an index after changing these settings.

## Python bindings

The optional `pylssg` package provides GIL-free, OpenMP-parallel batch insertion and filtered batch search:

```bash
python -m pip install ./python
```

```python
from pylssg import LSSGIndex

index = LSSGIndex(max_elements=1000, dimension=128, M=16, ef_construction=128, space="l2")
index.add_items(base_vectors, base_label_sets, threads=8)
ids, distances = index.search(query_vectors, query_label_sets, k=10, ef=100,
                               filter="containment", threads=8)
```

See [`python/README.md`](python/README.md) for the complete API. The regular CMake build remains C++-only; use `-DLSSG_BUILD_PYTHON=ON` when pybind11 and Python development headers are available.

A 100,000-vector synthetic build/search example is available at [`example/random_lssg.py`](example/random_lssg.py).

## Datasets

### Dataset file format

- **Vectors (`.fvecs`)**: each record consists of a 4-byte integer dimension `d`, followed by `d` floating-point values (`4*d` bytes in total for the vector payload).
- **Labels (`.txt`)**: each line is one label set, represented by comma-separated integer labels. The label file contains one label set per base/query vector.
- **Ground truth (`.bin`)**: each query record contains a 4-byte result count followed by that many 4-byte vector IDs. `search_lssg.sh` generates this file when it is missing.

### Dataset sources and descriptions

| Dataset | Size | Dim | \|A\| | \|ℒ\| | Description |
|---|---:|---:|---:|---:|---|
| [SIFT](http://corpus-texmex.irisa.fr/) | 1,000,000 | 128 | 12 | 2,385 | Classic ANNS benchmark; synthetic Zipf labels |
| [GIST](http://corpus-texmex.irisa.fr/) | 1,000,000 | 960 | 12 | 2,385 | Classic ANNS benchmark; synthetic Zipf labels |
| [TripClick](https://tripdatabase.github.io/tripclick/) | 1,055,976 | 768 | 29 | 7,734 | Health-search click logs embedded by DPR |
| [LAION](https://laion.ai/blog/laion-400-open-dataset/) | 1,000,448 | 512 | 30 | 62,066 | Image-caption pairs embedded by CLIP |
| [YTB-Video](https://research.google.com/youtube8m/index.html) | 1,000,000 | 1,024 | 3,862 | 160,018 | YouTube-8M topic labels and video features |
| [YTB-Audio](https://research.google.com/youtube8m/index.html) | 5,000,000 | 128 | 3,862 | 498,335 | YouTube-8M topic labels and audio features |
| [Wikipedia](https://huggingface.co/datasets/maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2) | 5,000,000 | 384 | 237,417 | 184,298 | English paragraph embeddings with Wikidata labels |
| [YFCC-1M](https://big-ann-benchmarks.com/neurips23.html) | 1,000,000 | 192 | 181,931 | 636,356 | Big-ANN NeurIPS'23 image metadata benchmark |

## Baselines

Evaluated using their official open-source implementations with recommended parameters:

- **LSSG-IVF** / **LSSG-MinHash**: our two variants, differing only in similar label-set selection. Unless stated otherwise, both use 9 tiers, `m=16`, and `efConstruction=128`; IVF uses candidate budget `φ=5,000` and MinHash uses `(τ, β)=(64, 16)`.
- **ACORN** ([repository](https://github.com/guestrin-lab/ACORN)): predicate-agnostic index.
- **Packing** ([repository](https://github.com/SpaceIshtar/FilterGraph)): the optimal strategy to solve LFANNS.
- **RWalks** ([repository](https://github.com/anon-sigmod/RWalks/tree/main)): label-vector and hybrid-distance index.
- **UNG** ([repository](https://github.com/YZ-Cai/Unified-Navigating-Graph)): separate indices per label set with cross-group edges.
- **ELI** ([repository](https://github.com/mingyu-hkustgz/LabelANN)): frequent-itemset-based index sharing.
- **ANNS libraries and databases**: FAISS, VSAG, Milvus, PGVector, HNSW, and k-means IVF.

## Evaluation

### Run LSSG

The public benchmark scripts are the primary build and search entry points. They are intentionally editable rather than parameterized command-line wrappers: open each script and edit the paths and parameters at the beginning.

1. Edit and run [`example/build_lssg.sh`](example/build_lssg.sh). Set `LSSG_USE_MINHASH=ON` for MinHash or `OFF` for IVF.
2. Edit and run [`example/search_lssg.sh`](example/search_lssg.sh). Its `LSSG_USE_MINHASH` value must match the index. If the configured ground-truth file is missing, the script generates it automatically, then brute-force-validates the first three queries.

```bash
bash example/build_lssg.sh
bash example/search_lssg.sh
```

### Summary of experimental results

On the evaluated real-world datasets, LSSG provides high-recall filtered ANNS for equality, containment, and overlap queries. LSSG-MinHash reduces label-indexing overhead as the label space grows, while LSSG-IVF provides a tunable candidate budget. The complete evaluation results can be found in [the benchmarking repository](https://github.com/ziqiwww/lssg_benchmark).

## License

See [LICENSE](LICENSE).
