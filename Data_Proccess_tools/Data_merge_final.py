import numpy as np

# File paths
positive_path = r".\collected\Positiv_data\Geodashreeldata1augmented.npz"
negative_path = r".\collected\Negativ_data\Geodashreeldatanegativ1_reduced.npz"
output_path = r".\collected\Final_data\Geodashreelfinaldata.npz"  # File for training

data = np.load(positive_path, allow_pickle=True)
print("Number of samples in positive file:", len(data["data"]))
print("Label of first sample:", data["data"][0][1])

# Load positive and negative data
pos_data = np.load(positive_path, allow_pickle=True)["data"]
neg_data = np.load(negative_path, allow_pickle=True)["data"]

# Concatenate
all_data = np.concatenate([pos_data, neg_data], axis=0)

# Shuffle (optional but recommended)
np.random.seed(42)
np.random.shuffle(all_data)

# Save
np.savez(output_path, data=all_data)
print(f"✅ Data merged and saved at: {output_path}")
