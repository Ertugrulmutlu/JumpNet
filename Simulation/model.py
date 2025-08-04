import torch.nn as nn
from torchvision import models
import torch

class JumpNet(nn.Module):
    """
    CNN model for jump/no-jump binary classification and hold duration regression.
    Uses MobileNetV2 backbone.
    """
    def __init__(self):
        super(JumpNet, self).__init__()
        base_model = models.mobilenet_v2(pretrained=True)
        self.backbone = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.jump_head = nn.Linear(512, 1)
        self.hold_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return torch.sigmoid(self.jump_head(x)), self.hold_head(x)
