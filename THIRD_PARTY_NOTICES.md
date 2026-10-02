# Third-party notices

RailWave's communication backend is derived from DeepEP. Its MIT license and
copyright notice are retained in `LICENSE`. The `deep_ep` package and extension
names preserve compatibility with the upstream API.

Vendored fmt headers report version 12.1.0 (`FMT_VERSION=120100`). Their notices
and the license in `third-party/fmt/LICENSE` are retained.

CUDA, PyTorch, NCCL, NVSHMEM, NumPy, and build tools are external dependencies.
Their licenses and installation terms continue to apply.

The NVSHMEM-derived device helper retains its NVIDIA notice and license
reference in `csrc/kernels/legacy/ibgda_device.cuh`. Other upstream attributions
remain in the relevant source comments.
