import cv2
import numpy as np

video_path = r".\recordings\Geodash1_235252.mp4"
jump_idx_path = r".\collected\Positiv_data\Geodashreeldata1.npz"
output_npz_path = r".\collected\Negativ_data\Geodashreeldatanegativ1.npz"

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 🔧 Get jump frame indices from the 'data' field in .npz
jump_data = np.load(jump_idx_path, allow_pickle=True)
jump_indices = [int(row[5]) for row in jump_data["data"]]  # frame_index column

# Exclude ±2 frames around each jump
excluded = set()
for idx in jump_indices:
    for offset in range(-2, 3):
        excluded.add(idx + offset)

# Collect negative samples (frames not used in positive data)
data_entries = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx not in excluded:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0  # Normalize

        entry = [
            frame_rgb,                           # image
            0.0,                                 # label (float! negative)
            np.array([0.], dtype=np.float32),    # keys_raw → empty but matches structure
            np.array([0], dtype=np.float32),     # hold_duration unknown
            "press",                             # phase → fixed
            frame_idx                            # frame_index
        ]
        data_entries.append(entry)

    frame_idx += 1

cap.release()

data_entries = np.array(data_entries, dtype=object)
keys_mapping = np.array([["w"]], dtype=object)

np.savez_compressed(output_npz_path, data=data_entries, keys=keys_mapping)

print(f"✅ Negative data created: {output_npz_path}")
print(f"🔢 Number of negative frames: {len(data_entries)}")
