# HPC Barnes-Hut Galaxy Simulation

## Overview

This project implements a simulation simulates the dynamics of galaxy formation by calculating gravitational interactions between millions of celestial bodies. While direct summation (Brute Force) scales with (O(N<sup>2</sup>)), this implementation utilizes the Barnes-Hut algorithm to approximate long-range interactions via an octree, reducing complexity to $O(N \log N)$.

![Barnes-Hut Simulation Example](https://upload.wikimedia.org/wikipedia/commons/9/93/2D_Quad-Tree_partitioning_of_100_bodies.png)

This repository represents a high-performance modernization of a serial C++ implementation, optimized for modern HPC clusters. Key enhancements include OpenMP threading for tree construction and force calculation, SIMD (AVX-512) vectorization, and a data layout transformation from AoS to SoA.

## Key Features & Optimizations

### 1. High-Performance Algorithm

**Parallel Tree Construction:** The domain is partitioned into sub-cubes, allowing 8<sup>L</sup> independent subtrees to be constructed concurrently using OpenMP tasks.

**Parallel Tree Destruction:** Top-level subtrees are deleted in parallel to reduce overhead during the simulation teardown phase.

**Barnes-Hut Approximation:** Clusters sufficiently far away (determined by the MAC parameter) are treated as single point masses. Each particles interactions are computed by traversing the octree, in parallel.

**Structure of Arrays (SoA):** Particle data is stored in SoA format (separate arrays for X, Y, Z, Mass) to improve cache locality and enable efficient vector loads.

**SIMD Vectorization (AVX-512):** The critical force calculation kernel is vectorized using AVX-512 intrinsics. Interactions are collected in a thread-local buffer (fitting in L1 cache) and processed in batches to maximize throughput.

### 3. Verification

**Correctness Suite:** Includes a testing framework that compares the Barnes-Hut result against a Brute Force baseline (O(N<sup>2</sup>)) to ensure error remains within a defined tolerance.


## Performance Results

The optimizations yield significant speedups compared to the baseline serial implementation:

Total SpeedUp = 53x for suffienctly large problem sizes (1 million particles)

## Build Instructions

Create build directory:

```bash
mkdir build && cd build
cmake .. -DHPCLab_BUILD_TESTS=ON
make -j
```
The test can be run by executing ./build/testBarnesHut

## Usage

The simulation can be configured via Command Line Arguments or a `config.json` file.

### Command Line Arguments

Arguments passed to the executable override `config.json` defaults.

| Argument | Description                                                                                                      |
|----------|------------------------------------------------------------------------------------------------------------------|
| `-n <int>` | Number of particles to simulate.                                                                                 |
| `--scenario <name>` | Name of a specific scenario to load (e.g., from `scenarios.json`).                                               |
| `--random` | Force generation of random particles (default).                                                                  |
| `--tree-level <1-4>` | Depth of parallel tree decomposition.<br>1=8 subtrees, 2=64, 3=512, 4=4096. Higher levels reduce load imbalance. |
| `--simd-batches <int>` | Number of SIMD batches to buffer before processing (Buffer length = simd_batches * 8)                            |
| `--help` | Display help message.                                                                                            |
| `--list` | List available scenarios.                                                                                        |

The simd-batches parameter is limited to a maximum of 16 by default because the buffers are statically allocated using a const inline expression (to avoid dynamic containers).

If you want to test batch sizes larger than 16, edit the MAx_SIMD_WIDTH parameter in include/InteractionForce.h.

For the batch size study, you must update this value and recompile each time you change it so the buffer is created in the corect size.


### Example Run

Run a simulation with 100,000 particles, using tree decomposition level 3 and custom SIMD batching:

```bash
./barnesHut -n 100000 --tree-level 3 --simd-batches 12
```

### Configuration File (config.json)

You can adjust physical parameters and simulation settings in `config.json`:

``` c++
{
  "massRange": 1000.0,        // Range for random mass generation [kg]
  "positionRange": 1000.0,    // Domain size [m]
  "velocityRange": 10.0,      // Initial velocity dispersion [m/s]
  "timeStep": 0.1,            // Simulation dt [s]
  "numParticles": 10000,      // Default particle count
  "barnesHutAlgorithm": true, // Toggle between BH and BruteForce
  "MAC": 0.5                  // Multipole Acceptance Criterion (theta)
}
```

## Visualization

A Python script is provided to visualize a 2D projection from the output CSV files.

**Requirements:**

```bash
pip install ffmpeg numpy pandas imageio
```

**Running the visualization:**

Ensure the simulation has generated output files (e.g., `particle_positions.csv`), then run the visualization script "3d_pyhton_plot.py" provided in the directory.

## Authors & Acknowledgments

### HPC Optimization & Parallelization:

- Christian Bolea-Schaser 
- Samet Kocbay 

### Original Implementation:

Developed as part of the Advanced Programming course at TUM by:

- Fabiana Lotter
- Samet Kocbay
- Sonja Lind

### Course Context:

This project was conducted for the Lab Course: Scientific Computing / High-Performance Computing at the Chair of Scientific Computing in Computer Science (SCCS), TUM under the supervision of Manish Mishra and Jonas Schuhmacher.
