import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class PatchCoreBackbone(nn.Module):
    def __init__(self, layers=["layer2", "layer3"]):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.output_layers = layers

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if "layer1" in self.output_layers:
            outputs.append(x)
        x = self.layer2(x)
        if "layer2" in self.output_layers:
            outputs.append(x)
        x = self.layer3(x)
        if "layer3" in self.output_layers:
            outputs.append(x)
        return outputs
