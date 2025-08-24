import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

    def forward(self, x):
        x = self.backbone(x)
        return x  # [B, C, H, W]

class FastFlow(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.flow = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 1)
        )

    def forward(self, x):
        z = self.flow(x)
        log_prob = -0.5 * torch.sum(z ** 2, dim=1)  # Gaussian log-likelihood
        return log_prob, z
