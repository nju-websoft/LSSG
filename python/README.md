# PyLSSG

`pylssg` provides Python bindings for the header-only LSSG index. Batch insertion and batch search release the Python GIL and use OpenMP, so both operations can run on multiple native threads.

## Install

From the repository root:

```bash
python -m pip install ./python
```

The default build uses the host's native SIMD features, matching the main CMake build. Use a portable build with:

```bash
LSSG_ENABLE_NATIVE_SIMD=OFF python -m pip install ./python
```

OpenMP is required for multi-threaded operation. Set `LSSG_ENABLE_NATIVE_SIMD=OFF` when building a wheel for machines with different CPU features.

## Usage

```python
import numpy as np
from pylssg import LSSGIndex

base = np.random.random((1000, 32)).astype(np.float32)
base_labels = [[i % 8, (i + 1) % 8] for i in range(len(base))]

index = LSSGIndex(
    max_elements=len(base),
    dimension=base.shape[1],
    M=16,
    ef_construction=128,
    space="l2",
)
index.add_items(base, base_labels, threads=8)
index.save("index/base.poi", "index/base.sco")

queries = base[:16]
query_labels = base_labels[:16]
ids, distances = index.search(
    queries,
    query_labels,
    k=10,
    ef=100,
    filter="containment",  # equality, containment, or overlap
    threads=8,
)
print(ids.shape, distances.shape)  # (16, 10), (16, 10)

loaded = LSSGIndex.load("index/base.poi", "index/base.sco", space="l2")
```

`ids` and `distances` are NumPy arrays with shape `(number_of_queries, k)`. An ID of `-1` and an infinite distance indicate that fewer than `k` matching vectors exist.

`label_sets` and `query_labels` are sequences of integer label sequences. Labels are sorted and deduplicated by the binding before they reach LSSG. `start_id` controls the first external vector ID in `add_items`; `threads=0` uses the OpenMP default.

The optional `label_file` constructor argument can preload a base label file in LSSG's comma-separated text format before insertion:

```python
index = LSSGIndex(1000, 32, label_file="base_labels.txt")
```

For CMake users, configure the optional module with `-DLSSG_BUILD_PYTHON=ON`. This requires a discoverable pybind11 CMake package and Python development headers; the normal `cmake .. && make -j` build does not require Python.

## Random-data example

The [`example/random_lssg.py`](../example/random_lssg.py) example builds 100,000 random 128-dimensional vectors and random label sets, then runs containment, equality, and overlap searches with OpenMP threads:

```bash
python example/random_lssg.py
```
