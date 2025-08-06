# G-LeCo: A GPU-Accelerated Integer Compression Library

G-LeCo is a high-performance, CUDA-based library designed for the compression and decompression of integer data streams. It leverages the massive parallelism of modern NVIDIA GPUs to accelerate every stage of the compression pipeline, from data partitioning and model fitting to random-access queries and full-file decompression.

The library is implemented as a header-only class (`G-LeCo.cuh`), making it easy to integrate into existing projects.

## Key Features

- **GPU-Accelerated End-to-End**: Utilizes CUDA for compression, decompression, serialization, and partitioning.
- **Flexible Partitioning**: Supports both fixed-length partitioning for simplicity and high-speed variable-length partitioning (using either CPU or GPU) to optimize compression ratios for varied data patterns.
- **High-Throughput Decompression**: Offers multiple decompression modes, including a "pre-unpacked" mode that trades higher memory usage for extremely fast random access and full-file decompression.
- **Optimized Random Access**: Includes highly specialized CUDA kernels for both generic and fixed-partition random access, minimizing latency for data queries.
- **Work-Stealing Load Balancing**: Employs advanced work-stealing kernels for full-file decompression on variably-sized partitions, ensuring optimal GPU utilization.

## Project Structure

The project is organized into functional modules to ensure clarity and maintainability.

```
G-LeCo_Project/
|
├── api/
│   ├── G-LeCo.cuh             # Main header-only class implementation
│   └── G-LeCo_Types.cuh       # Public data structures
|
├── core/
│   ├── InternalTypes.cuh      # Internal data structures
│   ├── CudaUtils.cuh          # CUDA macros and constants
│   ├── MathHelpers.cuh        # Host/Device math helpers
│   └── BitManipulation.cuh    # Bit extraction device functions
|
├── compression/
│   ├── CpuPartitioner.cuh     # CPU-based variable partitioner
│   ├── GpuPartitioner.cuh     # GPU-based variable partitioner
│   ├── PartitioningKernels.cu # Kernels for data partitioning
│   └── CompressionKernels.cu  # Kernels for model fitting and packing
|
├── decompression/
│   ├── FullDecompressionKernels.cu
│   ├── RandomAccessKernels.cu
│   ├── FixedPartitionKernels.cu
│   ├── WorkStealingDecompressionKernels.cu
│   └── UnpackKernel.cu
|
├── io/
│   ├── SerializationKernels.cu # Kernels for GPU serialization
│   └── FileUtils.cuh          # Checksum and file helpers
|
└── app/
    ├── BenchmarkUtils.cuh     # Benchmarking and file I/O utilities
    └── Main.cu                # Main application for testing
```

## Prerequisites

To compile and run this project, you will need:

- **NVIDIA CUDA Toolkit**: Version 11.0 or newer.
- **A C++11 compatible compiler**: Such as `g++`.
- **An NVIDIA GPU**: With Compute Capability 7.0 (Volta) or newer is recommended.

## Compilation

A `Makefile` is provided to simplify the compilation process. The project uses relocatable device code (`-rdc=true`) to link multiple `.cu` files.

1. **Clone the repository** and navigate to the project root directory.

2. **Build the application**: Open your terminal and run the `make` command.

   ```
   make
   ```

   This will compile all source files and create an executable named `g-leco-app` in the project's root directory.

3. **Clean build files**: To remove all compiled object files and the executable, run:

   ```
   make clean
   ```

## How to Run

The compiled `g-leco-app` executable is a benchmark tool that can run on synthetic or real data.

### Command-Line Syntax

```
./g-leco-app <data_type> [file_options]
```

- **`<data_type>`** (Required): Specifies the integer type to test.
  - Supported types: `int`, `long`, `long_long`, `unsigned_long_long` (or `ull`), `unsigned_int` (or `uint`).
- **`[file_options]`** (Optional): If provided, the application will load data from a file. If omitted, it will generate synthetic data.
  - `--text <filename>`: Load data from a plain text file (one number per line).
  - `--binary <filename>`: Load data from a binary file.

### Examples

1. **Run with synthetic `int` data:**

   ```
   ./g-leco-app int
   ```

2. **Run with synthetic `unsigned long long` data:**

   ```
   ./g-leco-app unsigned_long_long
   ```

3. **Run with data from a binary file named `my_data.bin` containing `long` integers:**

   ```
   ./g-leco-app long --binary my_data.bin
   ```

4. **Run with data from a text file named `timestamps.txt` containing `long_long` integers:**

   ```
   ./g-leco-app long_long --text timestamps.txt
   ```

## Example Scripts

To make running tests even easier, you can save the following content as shell scripts (e.g., `run_all_tests.sh`) and execute them. Make sure to give them execute permissions first (`chmod +x <script_name>.sh`).

### `build_and_run_synthetic.sh`

This script first cleans and builds the project, then runs benchmarks on synthetic data for all supported integer types.

```
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================="
echo "  Building G-LeCo Project"
echo "========================================="
# Clean previous builds and compile the project
make clean
make

echo ""
echo "========================================="
echo "  Running Benchmarks on Synthetic Data"
echo "========================================="

echo ""
echo "--- Testing 'int' type ---"
./g-leco-app int

echo ""
echo "--- Testing 'long' type ---"
./g-leco-app long

echo ""
echo "--- Testing 'long_long' type ---"
./g-leco-app long_long

echo ""
echo "--- Testing 'unsigned_int' type ---"
./g-leco-app unsigned_int

echo ""
echo "--- Testing 'unsigned_long_long' type ---"
./g-leco-app unsigned_long_long

echo ""
echo "========================================="
echo "  All synthetic data tests completed."
echo "========================================="
```

### `run_with_file.sh`

This script demonstrates how to run the benchmark with an external data file. You need to create a sample data file first.

```
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the filename and data type for the test
DATA_FILE="sample_data.txt"
DATA_TYPE="long_long"

# Check if the executable exists, build if not
if [ ! -f "g-leco-app" ]; then
    echo "Executable not found. Building project..."
    make
fi

# Create a sample text file for testing
echo "Creating a sample data file: ${DATA_FILE}"
echo "10000000000" > ${DATA_FILE}
echo "10000000050" >> ${DATA_FILE}
echo "10000000098" >> ${DATA_FILE}
echo "10000000145" >> ${DATA_FILE}
echo "10000000201" >> ${DATA_FILE}
for i in {1..1000}; do
    echo $((10000000201 + i * 50 + RANDOM % 20 - 10)) >> ${DATA_FILE}
done

echo ""
echo "====================================================="
echo "  Running Benchmark with data from '${DATA_FILE}'"
echo "  Data Type: ${DATA_TYPE}"
echo "====================================================="

./g-leco-app ${DATA_TYPE} --text ${DATA_FILE}

# Clean up the created sample file
rm ${DATA_FILE}

echo ""
echo "========================================="
echo "  File test completed."
echo "========================================="
```