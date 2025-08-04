import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error

def get_loss_functions():
    """
    Returns (binary classification loss, regression loss).
    """
    return nn.BCELoss(), nn.MSELoss()

def get_optimizer(model, lr=1e-4):
    """
    Returns Adam optimizer for given model.
    """
    return optim.Adam(model.parameters(), lr=lr)

def save_model(model, path="jumpnet_model.pt"):
    """
    Saves model weights to the specified path.
    """
    torch.save(model.state_dict(), path)

def load_model(model_class, path="jumpnet_model.pt", device="cpu"):
    """
    Loads model weights and returns the model on given device.
    """
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def evaluate(model, dataloader, device):
    """
    Evaluates the model and returns a dictionary of metrics.
    """
    model.eval()
    all_labels, all_preds = [], []
    hold_labels, hold_preds = [], []

    with torch.no_grad():
        for images, jump_labels, hold_durations in dataloader:
            images, jump_labels, hold_durations = images.to(device), jump_labels.to(device), hold_durations.to(device)
            jump_preds, hold_preds_batch = model(images)
            jump_preds_bin = (jump_preds > 0.5).float()

            all_labels.extend(jump_labels.cpu().numpy())
            all_preds.extend(jump_preds_bin.cpu().numpy())

            mask = (jump_labels > 0.5).squeeze()
            if mask.any():
                hold_labels.extend(hold_durations[mask].cpu().numpy())
                hold_preds.extend(hold_preds_batch[mask].cpu().numpy())

    return {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "F1 Score": f1_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds),
        "Recall": recall_score(all_labels, all_preds),
        "Hold Duration MSE": mean_squared_error(hold_labels, hold_preds)
    }
