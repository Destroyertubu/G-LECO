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
