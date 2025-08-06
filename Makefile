# Makefile for the G-LeCo Project

# Compiler and Flags
# =============================================================================
NVCC := nvcc
CXX := g++

# General Flags
CXXFLAGS := -O3 -Wall -std=c++11
NVCCFLAGS := -O3 -std=c++11
LDFLAGS := 

# CUDA Specific Flags
# Relocatable Device Code is required to link multiple .cu files
NVCCFLAGS += -rdc=true
# Specify the target GPU architecture. Change 'sm_75' to match your GPU.
# Examples: sm_70 (Volta), sm_80 (Ampere), sm_86 (Ampere), sm_90 (Hopper)
NVCCFLAGS += -arch=sm_75
# Link against the CUB library (often included with CUDA Toolkit)
NVCCFLAGS += -lcub


# Directories
# =============================================================================
# Source Directories
API_DIR := api
CORE_DIR := core
COMP_DIR := compression
DECOMP_DIR := decompression
IO_DIR := io
APP_DIR := app

# Build Directory for Object Files
BUILD_DIR := build


# Source Files and Object Files
# =============================================================================
# Find all .cu source files in the specified directories
# Note: api/G-LeCo.cu is not included as it's a header-only implementation now.
CU_SOURCES := $(wildcard $(COMP_DIR)/*.cu) \
              $(wildcard $(DECOMP_DIR)/*.cu) \
              $(wildcard $(IO_DIR)/*.cu) \
              $(wildcard $(APP_DIR)/*.cu)

# Generate corresponding object file paths in the build directory
OBJECTS := $(patsubst %.cu, $(BUILD_DIR)/%.o, $(CU_SOURCES))

# Executable Target Name
TARGET := g-leco-app


# Build Rules
# =============================================================================
# Default target: build the application
all: $(TARGET)

# Rule to link the final executable
$(TARGET): $(OBJECTS)
	@echo "==> Linking executable: $@"
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "==> Build complete. Executable created: ./$(TARGET)"

# Rule to compile .cu files into object files (.o)
# -I. allows includes to be relative to the project root (e.g., #include "api/G-LeCo.cuh")
$(BUILD_DIR)/%.o: %.cu
	@echo "==> Compiling: $<"
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -I. -c $< -o $@

# Clean target
clean:
	@echo "==> Cleaning build files..."
	@rm -f $(TARGET)
	@rm -rf $(BUILD_DIR)
	@echo "==> Clean complete."

# Phony targets
.PHONY: all clean
