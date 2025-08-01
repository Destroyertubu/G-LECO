import numpy as np
from scipy.stats import norm
import os

# --- Configuration Parameters ---
NUM_KEYS = 200000000
TARGET_DTYPE = np.uint32

# Set a fixed random seed for reproducible results
np.random.seed(seed=42)

def generate_and_save_data(filename, data_label, transform_function):
    """
    Generates, transforms, normalizes, and saves data to a file.

    Args:
        filename (str): The name of the output file.
        data_label (str): A label for the data being generated (e.g., "linear", "normal") for logging.
        transform_function (callable): A function to apply to the initial linear data.
    """
    print(f"Generating {data_label} data...")
    if os.path.exists(filename):
        print(f"File '{filename}' already exists, skipping generation.")
        return

    print(f"Data type: {TARGET_DTYPE.__name__}...")

    # 1. Create a standardized linear space as a base
    # Use float64 to ensure precision
    base_keys = np.linspace(0, 1, NUM_KEYS + 2, dtype=np.float64)[1:-1]

    # 2. Apply the specified distribution transformation function
    transformed_keys = transform_function(base_keys)

    # 3. Normalize the data to a [0, 1] range
    min_val, max_val = np.min(transformed_keys), np.max(transformed_keys)
    if min_val < max_val:
        normalized_keys = (transformed_keys - min_val) / (max_val - min_val)
    else:
        normalized_keys = np.zeros_like(transformed_keys) # All zeros if all values are the same

    # 4. Scale to the maximum value of the target integer type and cast the type
    scaled_keys = (normalized_keys * np.iinfo(TARGET_DTYPE).max).astype(TARGET_DTYPE)

    # 5. Save to file
    np.savetxt(filename, scaled_keys, fmt='%d')
    print(f"Successfully generated file: '{filename}'")

# --- Transformation Functions for Different Distributions ---

def linear_transform(keys):
    """A linear transformation (i.e., does nothing)."""
    return keys

def normal_transform(keys):
    """
    Applies the normal distribution's percent point function (PPF).
    This is processed in chunks due to the high memory usage of the PPF function.
    """
    # Split the data into 1000 chunks to reduce memory pressure per operation
    key_chunks = np.array_split(keys, 1000)
    
    # Apply PPF to each chunk and concatenate the results
    ppf_chunks = [norm.ppf(chunk) for chunk in key_chunks]
    return np.concatenate(ppf_chunks)

# --- Main Program Entry ---
if __name__ == "__main__":
    # Generate linear data
    generate_and_save_data(
        filename=f"linear_{NUM_KEYS // 1000000}M_uint32.txt",
        data_label="linear",
        transform_function=linear_transform
    )

    print("-" * 30) # Add a separator for log clarity

    # Generate normal distribution data
    generate_and_save_data(
        filename=f"normal_{NUM_KEYS // 1000000}M_uint32.txt",
        data_label="normal",
        transform_function=normal_transform
    )
