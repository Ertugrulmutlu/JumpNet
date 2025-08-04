import numpy as np
import cv2
import random
from tqdm import tqdm

input_path = r".\collected\Positiv_data\Geodashreelmergeddata.npz"
output_path = r".\collected\Positiv_data\Geodashreeldata1augmented.npz"
augmentation_multiplier = 2  # For each sample, create 2 new ones (total x3)

data = np.load(input_path, allow_pickle=True)
entries = data["data"]
keys = data["keys"]

augmented = []

def augment_image(img):
    """Apply random augmentations to a given image."""
    aug_img = img.copy()

    if random.random() < 0.5:
        aug_img = cv2.flip(aug_img, 1)  # Horizontal flip

    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        aug_img = np.clip(aug_img * factor, 0, 1)  # Brightness adjustment

    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, aug_img.shape)
        aug_img = np.clip(aug_img + noise, 0, 1)  # Add Gaussian noise

    if random.random() < 0.3:
        M = np.float32([[1, 0, random.randint(-5, 5)], [0, 1, 0]])
        aug_img = cv2.warpAffine(aug_img, M, (aug_img.shape[1], aug_img.shape[0]))  # Horizontal shift

    return aug_img.astype(np.float32)

# Add original + augmented samples
final_data = list(entries)

for entry in tqdm(entries, desc="Augmenting"):
    for _ in range(augmentation_multiplier):
        new_entry = entry.copy()
        new_img = augment_image(entry[0])
        new_entry[0] = new_img
        final_data.append(new_entry)

# Save
final_data = np.array(final_data, dtype=object)
np.savez_compressed(output_path, data=final_data, keys=keys)

print(f"✅ Augmented positive data created: {output_path}")
print(f"🔢 Total number of samples: {len(final_data)}")
