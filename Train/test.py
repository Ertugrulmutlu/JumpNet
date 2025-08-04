import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import load_dataset
from src.model import JumpNet
from src.utils import evaluate, load_model

# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define the same transform as used in training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((227, 227)),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 3. Load your test dataset (adjust path!)
_, test_dataset = load_dataset(r".\datas\Geodashreelfinaldata.npz", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# 4. Load trained model
model = load_model(JumpNet, "jumpnet_model.pt", device)

# 5. Evaluate the model
metrics = evaluate(model, test_loader, device)
print("\n=== Model Evaluation Metrics ===")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
