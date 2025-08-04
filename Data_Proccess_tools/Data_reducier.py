import numpy as np

input_path = r".\collected\Negativ_data\Geodashreeldatanegativ1.npz"
output_path = r".\collected\Negativ_data\Geodashreeldatanegativ1_reduced.npz"
keep_count = 500  # Number of samples to randomly keep

data = np.load(input_path, allow_pickle=True)
all_entries = data["data"]
keys = data["keys"]

total = len(all_entries)
keep_count = min(keep_count, total)  # If dataset is small, keep all

np.random.seed(42)  # Fixed seed for reproducibility
indices = np.random.choice(total, keep_count, replace=False)
reduced_entries = all_entries[sorted(indices)]  # Sorted for viewer compatibility

# Save as a new file
np.savez_compressed(output_path, data=reduced_entries, keys=keys)

print(f"✅ Negative .npz file reduced to {keep_count} samples and saved as: {output_path}")
