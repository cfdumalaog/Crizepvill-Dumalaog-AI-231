"""TinyDSCNN-48 model architecture with random initialization."""
import torch
from torch import nn
from .config import CLASSES, CHANNELS, DROPOUT


class TinyDSCNN(nn.Module):
    def __init__(self, classes=CLASSES, channels=CHANNELS, dropout=DROPOUT):
        super().__init__()
        layers = [
            nn.Conv2d(1, channels, (5, 5), stride=(2, 2), padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ]
        for i in range(4):
            layers.extend([
                nn.Conv2d(channels, channels, 3, stride=(1, 2) if i == 1 else 1,
                          padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ])
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels, classes)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        dropped = self.dropout(pooled)
        return self.classifier(dropped)
