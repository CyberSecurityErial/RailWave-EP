# RailWave-EP

RailWave balances traffic across GPU–NIC rails and schedules inter-node
transfers in permutation waves. The communication backend is based on DeepEP;
its Python interface remains `deep_ep.ElasticBuffer`.

Rail balancing and permutation scheduling compose four paths:

| Path | Rail balancing | Permutation waves |
| --- | --- | --- |
| Native | Off | Off |
| Rail | On | Off |
| Permutation | Off | On |
| Joint | On | On |

`railwave.decide` selects a path from caller-supplied calibration anchors and
defaults to Joint outside their operating regions.

## Build

GPU execution requires Linux, CUDA, PyTorch, NCCL with Gin support, and NVSHMEM.
The build currently includes DeepEP's legacy kernels, so NVSHMEM is required
even when only the elastic interface is used. fmt headers are vendored.

Set the dependency roots and install from the repository:

```bash
export CUDA_HOME=...
export EP_NCCL_ROOT_DIR=...
export EP_NVSHMEM_ROOT_DIR=...
export TORCH_CUDA_ARCH_LIST=9.0
export LIBRARY_PATH="$EP_NCCL_ROOT_DIR/lib:$EP_NVSHMEM_ROOT_DIR/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$EP_NCCL_ROOT_DIR/lib:$EP_NVSHMEM_ROOT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install --no-build-isolation .
```

Each communication dependency root must contain `include/` and `lib/`.
PyTorch and the extension must load the same NCCL library. Runtime kernel
compilation requires `nvcc` and a writable `EP_JIT_CACHE_DIR`.

## Use the scheduler

```python
from railwave import plan_cyclic_source_local_waves

demand = [[0, 5, 3, 2], [2, 0, 5, 3], [3, 2, 0, 5], [5, 3, 2, 0]]
waves = plan_cyclic_source_local_waves(demand)
```

Each transfer is `(source, destination, offset, count, edge_total)`. Within a
wave, each node sends to at most one destination and receives from at most one
source. The executor completes every wave in order. The optional equal-chunk
planner splits large edges and sends their residual tails last.

## Layout

- `railwave/`: CPU schedulers and policy lookup.
- `deep_ep/`, `csrc/`: communication interface and CUDA implementation.
- `third-party/fmt/`: formatting headers used by the build.

Upstream licenses and attributions are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
