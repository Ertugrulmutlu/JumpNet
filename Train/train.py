import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from src.dataset import load_dataset
from src.model import JumpNet
from src.utils import get_loss_functions, get_optimizer, save_model

# Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# 🖼 Image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((227, 227)),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load dataset (train-test split is automatic)
train_dataset, test_dataset = load_dataset(r".\\datas\\Geodashreelfinaldata.npz", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model, loss functions and optimizer
model = JumpNet().to(device)
criterion_cls, criterion_reg = get_loss_functions()
optimizer = get_optimizer(model)

# TensorBoard logging
writer = SummaryWriter("runs/jump_training")

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    for batch_idx, (images, jump_labels, hold_durations) in enumerate(train_loader):
        images, jump_labels, hold_durations = images.to(device), jump_labels.to(device), hold_durations.to(device)

        optimizer.zero_grad()
        jump_preds, hold_preds = model(images)

        #  Binary classification loss
        loss_cls = criterion_cls(jump_preds, jump_labels)

        #  Regression loss (only for positive samples)
        mask = jump_labels > 0.5
        if mask.any():
            loss_reg = criterion_reg(hold_preds[mask], hold_durations[mask])
        else:
            loss_reg = torch.tensor(0.0).to(device)

        loss = loss_cls + loss_reg
        loss.backward()
        optimizer.step()

        # Totals for tracking
        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_reg_loss += loss_reg.item()

        # Batch output (every 10 steps)
        if batch_idx % 10 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} (Cls: {loss_cls.item():.4f}, Reg: {loss_reg.item():.4f})")

    # Print epoch average losses and log to TensorBoard
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)

    writer.add_scalar("Loss/Total", avg_loss, epoch)
    writer.add_scalar("Loss/Classification", avg_cls_loss, epoch)
    writer.add_scalar("Loss/Regression", avg_reg_loss, epoch)

    print(f"\n✅ Epoch {epoch+1} completed | Total: {avg_loss:.4f} | Cls: {avg_cls_loss:.4f} | Reg: {avg_reg_loss:.4f}\n")

# Save trained model
save_model(model)
writer.close()
