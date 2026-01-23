# CUDA NDT Matcher

A CUDA/Rust implementation of Autoware's NDT scan matcher for autonomous vehicle localization.

## Overview

Re-implements `autoware_ndt_scan_matcher` using:
- **CubeCL** for GPU compute kernels
- **CUB** for high-performance primitives (radix sort, segmented reduce)
- **Persistent kernel** architecture with in-kernel Jacobi SVD solver

Drop-in replacement for Autoware's NDT with identical ROS 2 interface.

## Prerequisites

- ROS 2 Humble (`/opt/ros/humble`)
- Autoware 1.5.0 (`/opt/autoware/1.5.0`)
- NVIDIA GPU with CUDA support
- Rust toolchain (`rustup`)
- [direnv](https://direnv.net/) (recommended)

## Setup

```bash
# Allow direnv to load environment
direnv allow

# Check prerequisites and install build dependencies
just setup
```

This will verify:
- Rust toolchain is installed
- ROS Humble is available at `/opt/ros/humble`
- Autoware 1.5.0 is installed at `/opt/autoware/1.5.0`

## Build

```bash
# Build all packages
just build
```

## Run

```bash
# Download sample data (one-time)
just download-data

# Extract rosbags
cd data && ./extract.sh && cd ..

# Run CUDA NDT demo
just run-cuda
```

## Debug

Enable per-alignment debug output:

```bash
# Run with debug logging
just run-cuda-debug

# Analyze results
just analyze-debug-cuda
```

Debug output is written to `logs/ndt_cuda_debug.jsonl`.

### Environment Variables

| Variable         | Description                                        |
|------------------|----------------------------------------------------|
| `NDT_DEBUG=1`    | Enable debug output                                |
| `NDT_DEBUG_FILE` | Output path (default: `logs/ndt_cuda_debug.jsonl`) |
| `NDT_USE_GPU=0`  | Force CPU mode                                     |

## Profile

Compare against Autoware's builtin NDT:

```bash
# Build patched Autoware for comparison
just build-comparison

# Run Autoware builtin with debug
just run-builtin-debug

# Analyze Autoware results
just analyze-debug-autoware
```

Results are documented in `docs/profiling-results.md`.

## Project Structure

```
src/
  ndt_cuda/              # CubeCL NDT library
  cuda_ffi/              # CUDA FFI bindings (CUB)
  cuda_ndt_matcher/      # ROS 2 node
  cuda_ndt_matcher_launch/  # Launch files
```

## License

Apache 2.0
