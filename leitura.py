import numpy as np

# Load the dataset
data = np.load("pendulo_critico.npz")

# Show available arrays
keys = list(data.keys())

# Build a summary
summary = {}
for k in keys:
    arr = data[k]
    summary[k] = {
        "shape": arr.shape,
        "dtype": arr.dtype,
        "min": float(arr.min()) if arr.ndim > 0 else arr,
        "max": float(arr.max()) if arr.ndim > 0 else arr,
    }

keys, summary
