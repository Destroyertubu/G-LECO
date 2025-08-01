# G-LeCo-1

This repository contains three distinct integer compression frameworks.

1. **G-LeCo (Our Method)**: Our GPU-accelerated framework using learned models.
2. **TILE**: A GPU implementation of classic block-based compression, based on the paper *"Tile-based Lightweight Integer Compression in GPU"* (Shanbhag et al., SIGMOD '22).
3. **FastLanes**: A CPU library with a cache-friendly memory layout, based on the paper *"The FastLanes Compression Layout"* (Afroozeh & Boncz, PVLDB 2023).

## 1. G-LeCo: GPU Learned Compression

- **Source**: `G-LeCo.cu`

### Compilation

```
nvcc -O3 -std=c++17 -arch=sm_70 G-LeCo.cu -o gleco_benchmark
```

*Note: Adjust `-arch=sm_70` for your GPU (e.g., Turing: `sm_75`, Ampere: `sm_86`).*

### Usage

**Syntax:**

```
./gleco_benchmark <data_type> [--text | --binary] [input_filename]
```

- **Example (Synthetic Data):**

  ```
  ./gleco_benchmark long_long
  ```

- **Example (Binary File):**

  ```
  ./gleco_benchmark ull --binary my_ull_data.bin
  ```

## 2. TILE: GPU Parallel Compression

- **Source**: `tile.cu`

### Compilation

```
nvcc -O3 -std=c++14 -arch=sm_70 tile.cu -o tile_benchmark
```

*Note: Adjust `-arch=sm_70` for your GPU.*

### Usage

**Syntax:**

```
./tile_benchmark --dtype <type> --algo <algo> --input <file> [--output <file>] [--size <num>]
```

- **Example (Compress File):**

  ```
  ./tile_benchmark --dtype int --algo DFOR --input my_int_data.bin
  ```

- **Example (Synthetic Data):**

  ```
  ./tile_benchmark --dtype ulonglong --algo ALL --size 1000000
  ```

## 3. FastLanes: CPU Scalar Compression

- **Source**: `fastlanes_benchmark.cpp`

### Compilation

```
g++ -O3 -std=c++17 -fopenmp fastlanes_benchmark.cpp -o fastlanes_benchmark
```

### Usage

**Syntax:**

```
./fastlanes_benchmark <data_type> [--text | --binary] <filename>
```

- **Example (Text File):**

  ```
  ./fastlanes_benchmark int --text my_int_data.txt
  ```

## Dependencies

- **GPU Code (`.cu`)**:
  - NVIDIA CUDA Toolkit (12.0+)
  - GPU with Compute Capability 7.0+
- **CPU Code (`.cpp`)**:
  - C++17 Compiler (GCC, Clang)
  - OpenMP (Recommended)
