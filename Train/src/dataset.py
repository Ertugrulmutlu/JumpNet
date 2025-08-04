import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image

class JumpDataset(Dataset):
    """
    Custom PyTorch Dataset for jump action data.
    """
    def __init__(self, entries, transform=None):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image_np = self.entries[idx][0]
        # Ensure image is uint8 before converting to PIL
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image_np)

        label = float(self.entries[idx][1])
        hold_duration = float(self.entries[idx][3])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32), torch.tensor([hold_duration], dtype=torch.float32)

def load_dataset(npz_path, transform=None, split_ratio=0.8):
    """
    Loads dataset from .npz and returns (train_dataset, test_dataset).
    """
    data = np.load(npz_path, allow_pickle=True)
    entries = data["data"]
    total_len = len(entries)
    train_len = int(total_len * split_ratio)
    test_len = total_len - train_len
    train_data, test_data = random_split(entries, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    return JumpDataset(train_data, transform), JumpDataset(test_data, transform)
