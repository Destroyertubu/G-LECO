# G-LECO-1

This repository contains implementations of three distinct, high-performance integer compression frameworks. Each library showcases a different approach to data compression, targeting various hardware and performance goals, from GPU-accelerated learned models to advanced CPU memory layouts.

## Repository Overview

This collection features three standalone compression libraries:

1. **G-LeCo (`G-LeCo.cu`)**: A GPU-accelerated framework that uses learned models to predict and compress data.
2. **GPCC (`tile.cu`)**: A GPU-based implementation of classic, block-based compression algorithms.
3. **FastLanes (`fastlanes_benchmark.cpp`)**: A CPU-based library that uses an innovative memory layout to achieve extremely fast decompression with scalar code.

## 1. G-LeCo: GPU-Accelerated Learned Compression

- **Source File**: `G-LeCo.cu`

### Compilation

To compile the G-LeCo framework, you will need the NVIDIA CUDA Toolkit (12+) installed. Use the following command in your terminal:

```
nvcc -O3 -std=c++17 -arch=sm_70 G-LeCo.cu -o gleco_benchmark
```

**Note**: The `-arch=sm_70` flag is for GPUs with Volta architecture. You should adjust this to match your hardware (e.g., Turing: `sm_75`, Ampere: `sm_86`).

### Usage

The program can either generate its own synthetic data for testing or load data from a file.

**Syntax:**

```
./gleco_benchmark <data_type> [--text | --binary] [input_filename]
```

- `<data_type>`: The type of data to use. Supported options are `int`, `long`, `long_long`, `unsigned_long_long` (or `ull`), and `uint`.
- `[input_filename]`: The path to a data file. If not provided, synthetic data will be generated.
- `--text` or `--binary`: Specifies the format of the input file. This is optional and defaults to text.

**Examples:**

- **Run with synthetic data:**

  ```
  ./gleco_benchmark long_long
  ```

- **Run with data from a binary file:**

  ```
  ./gleco_benchmark ull --binary my_ull_data.bin
  ```

## 2. GPCC: GPU Parallel Columnar Compression

- **Source File**: `tile.cu`

### Introduction

This library provides GPU-accelerated versions of well-established, block-based integer compression techniques. It is designed for efficiency and high performance when compressing columnar data. The primary algorithms implemented are:

- **FOR (Frame-of-Reference)**
- **DFOR (Delta-of-Reference)**
- **RFOR (a hybrid of RLE and FOR)**

The implementation processes data in parallel blocks, making it well-suited for the CUDA architecture.

### Compilation

Ensure the NVIDIA CUDA Toolkit is installed and run the following command:

```
nvcc -O3 -std=c++14 -arch=sm_70 tile.cu -o gpcc_benchmark
```

**Note**: Adjust the `-arch=sm_70` flag as needed for your specific GPU architecture.

### Usage

The program offers a range of command-line flags to control its execution.

**Syntax:**

```
./gpcc_benchmark --dtype <type> --algo <algo> --input <file> [--output <file>]
```

- `--dtype <type>`: Specifies the data type (e.g., `int`, `uint`, `longlong`).
- `--algo <algo>`: Selects the compression algorithm (`FOR`, `DFOR`, `RFOR`, or `ALL` to test all).
- `--input <file>`: Path to the input data file.
- `--output <file>`: (Optional) Path to save the compressed output.
- `--size <number>`: (Optional) If no input file is provided, generates synthetic data of this size.

**Examples:**

- **Compress a file using DFOR:**

  ```
  ./gpcc_benchmark --dtype int --algo DFOR --input my_int_data.bin
  ```

- **Test all algorithms on synthetic data:**

  ```
  ./gpcc_benchmark --dtype ulonglong --algo ALL --size 1000000
  ```

## 3. FastLanes: CPU Scalar Compression Layout

- **Source File**: `fastlanes_benchmark.cpp`

### Introduction

This is a CPU-only implementation based on the principles of the "FastLanes Compression Layout" paper. Its main innovation is a special memory layout that rearranges data to be exceptionally friendly to modern CPU caches and prefetchers. This allows for extremely high-speed decompression using simple, scalar C++ code, without relying on explicit SIMD/AVX instructions. It demonstrates that optimizing data layout for hardware can yield significant performance gains.

### Compilation

This is a standard C++ file and can be compiled with `g++` or `clang++`. For best performance, enabling OpenMP is recommended.

```
g++ -O3 -std=c++17 -fopenmp fastlanes_benchmark.cpp -o fastlanes_benchmark
```

### Usage

The program is executed by specifying a data type and an input file.

**Syntax:**

```
./fastlanes_benchmark <data_type> [--text | --binary] <filename>
```

- `<data_type>`: The data type to be processed. Supported types include `int`, `unsigned_int`, and `unsigned_long_long`.
- `<filename>`: The path to the input data file.
- `--text` or `--binary`: (Optional) Specifies if the file is in text or binary format.

**Example:**

- **Process an integer text file:**

  ```
  ./fastlanes_benchmark int --text my_int_data.txt
  ```

## Dependencies

- **GPU Code** (`.cu` files):
  - NVIDIA CUDA Toolkit (11.0 or newer recommended).
  - An NVIDIA GPU with Compute Capability 7.0 (Volta) or higher.
- **CPU Code** (`.cpp` file):
  - A C++17 compliant compiler (e.g., GCC, Clang).
  - OpenMP is recommended for enabling multithreading.
