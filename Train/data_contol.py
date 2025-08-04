import numpy as np

# 🔧 Set the path to your dataset
npz_path = r".\\datas\\Geodashreelfinaldata.npz"  # Path to the merged dataset file

data = np.load(npz_path, allow_pickle=True)
entries = data["data"]

print(f"\n🧠 Total number of samples: {len(entries)}")

# Count
pos_count = sum(entry[1] == 1.0 for entry in entries)
neg_count = sum(entry[1] == 0.0 for entry in entries)
other_count = len(entries) - (pos_count + neg_count)

print(f"✅ Positive (jump) samples: {pos_count}")
print(f"❌ Negative (no-jump) samples: {neg_count}")
if other_count > 0:
    print(f"⚠️ Warning: {other_count} samples have corrupted labels (neither 1.0 nor 0.0)")

# Hold durations (only for positive samples)
hold_durations = [float(entry[3].flatten()[0]) for entry in entries if entry[1] == 1.0]

if hold_durations:
    print(f"\n🕒 Hold Duration statistics (jump duration):")
    print(f"  Min : {min(hold_durations):.4f}")
    print(f"  Max : {max(hold_durations):.4f}")
    print(f"  Mean: {np.mean(hold_durations):.4f}")
    print(f"  Std : {np.std(hold_durations):.4f}")
else:
    print("⚠️ No hold_duration found for positive samples!")

# Show sample data
print("\n🖼 Sample 1:")
sample = entries[0]
print(f"  image shape  : {sample[0].shape}")
print(f"  label        : {sample[1]}")
print(f"  hold_duration: {sample[3]}")
print(f"  phase        : {sample[4]}")
print(f"  frame_index  : {sample[5]}")
