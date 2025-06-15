import torch
from torchvision import models

import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)


class ResNet18(nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()
        freeze_params = config.get('freeze_params', True)
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = (
            nn.Identity()
        )  # Remove the final fully connected layer

        if freeze_params:
            print('Freezing ResNet18 parameters')
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, config):
        super(ResNet34, self).__init__()
        freeze_params = config.get('freeze_params', True)
        self.backbone = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.backbone.fc = (
            nn.Identity()
        )  # Remove the final fully connected layer

        if freeze_params:
            print('Freezing ResNet34 parameters')
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, config):
        super(ResNet50, self).__init__()
        freeze_params = config.get('freeze_params', True)
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = (
            nn.Identity()
        )  # Remove the final fully connected layer

        if freeze_params:
            print('Freezing ResNet50 parameters')
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)  # Example input tensor
    model = ResNet18()
    output = model(x)
    print(f'ResNet18 output shape: {output.shape}')
    model = ResNet34()
    output = model(x)
    print(f'ResNet34 output shape: {output.shape}')
    model = ResNet50()
    output = model(x)
    print(f'ResNet50 output shape: {output.shape}')

# ResNet18 output shape: torch.Size([1, 512])
# ResNet34 output shape: torch.Size([1, 512])
# ResNet50 output shape: torch.Size([1, 2048])
