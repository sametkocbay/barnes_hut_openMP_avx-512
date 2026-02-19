# Barnes-Hut N-Body Galaxy Simulation

## Overview

This project simulates the dynamics of galaxy formation by calculating gravitational interactions between celestial bodies. While direct summation (Brute Force) scales with O(N²), this implementation utilizes the **Barnes-Hut algorithm** to approximate long-range interactions via an octree, reducing complexity to O(N log N).

![Barnes-Hut Simulation Example](https://upload.wikimedia.org/wikipedia/commons/9/93/2D_Quad-Tree_partitioning_of_100_bodies.png)

The simulation is optimized for performance using **OpenMP** parallelism, optional **AVX-512 SIMD** vectorization, and a **Structure of Arrays (SoA)** data layout. It compiles and runs on any modern C++17 compiler — AVX-512 is used automatically when available, otherwise a scalar fallback is used.

## Key Features & Optimizations

- **Barnes-Hut Approximation:** Clusters sufficiently far away (determined by the MAC / theta parameter) are treated as single point masses, computed via parallel octree traversal.
- **Parallel Tree Construction & Destruction:** The domain is partitioned into sub-cubes, allowing 8^L independent subtrees to be constructed and destroyed concurrently using OpenMP tasks.
- **Structure of Arrays (SoA):** Particle data is stored in SoA format (separate arrays for X, Y, Z, Mass) to improve cache locality and enable efficient vector loads.
- **SIMD Vectorization (AVX-512):** When available, the force calculation kernel is vectorized using AVX-512 intrinsics with batched processing. On systems without AVX-512, a scalar fallback is used automatically.
- **Correctness Suite:** Includes a Google Test framework that compares Barnes-Hut results against a Brute Force baseline to ensure error remains within tolerance for various theta values.

## Build Instructions

### Requirements

- C++17 compiler (GCC, Clang, MSVC, etc.)
- CMake ≥ 3.25
- OpenMP support
- (Optional) AVX-512 capable CPU for SIMD acceleration

### Building

```bash
mkdir build && cd build
cmake ..
make -j
```

To build with unit tests:

```bash
cmake .. -DHPCLab_BUILD_TESTS=ON
make -j
```

To explicitly disable AVX-512 (e.g., for portability):

```bash
cmake .. -DUSE_AVX512=OFF
make -j
```

Tests can be run with `./testBarnesHut` from the build directory.

## Usage

The simulation can be configured via command line arguments or a `config.json` file.

### Command Line Arguments

Arguments passed to the executable override `config.json` defaults.

| Argument | Description |
|----------|-------------|
| `-n <int>` | Number of particles to simulate. |
| `--scenario <name>` | Name of a specific scenario to load from `scenarios.json`. |
| `--random` | Force generation of random particles (default). |
| `--tree-level <1-4>` | Depth of parallel tree decomposition (1=8, 2=64, 3=512, 4=4096 subtrees). |
| `--simd-batches <int>` | Number of SIMD batches to buffer before processing (buffer length = batches × 8). |
| `--help` | Display help message. |
| `--list` | List available scenarios. |

> **Note:** The `--simd-batches` parameter is limited to 16 by default (statically allocated buffers). To test larger batch sizes, edit `MAX_SIMD_BATCHES` in `include/InteractionForce.h` and recompile.

### Example

```bash
./barnesHut -n 100000 --tree-level 3 --simd-batches 12
```

### Configuration File (config.json)

```json
{
  "massRange": 1000.0,
  "positionRange": 1000.0,
  "velocityRange": 10.0,
  "timeStep": 0.1,
  "numParticles": 10000,
  "barnesHutAlgorithm": true,
  "MAC": 0.5
}
```

## About

This project originated as a university project at TUM. I was involved in both the original implementation and the subsequent HPC optimization and parallelization work.
