import numpy as np

file1 = r".\collected\Geodashreeldata1.npz"
file2 = r".\collected\Geodashreeldata2.npz"
output_file = r".\collected\Geodashreelmergeddata.npz"

data1 = np.load(file1, allow_pickle=True)
data2 = np.load(file2, allow_pickle=True)

# Merge the 'data' fields
merged_data = np.concatenate([data1["data"], data2["data"]])

# Take the 'keys' field once, make it unique
merged_keys = np.unique(np.concatenate([data1["keys"], data2["keys"]]))

# Save results
np.savez_compressed(output_file, data=merged_data, keys=merged_keys)

print("✅ Merge completed:", output_file)
print("🧩 Merged keys:", merged_keys)
