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
