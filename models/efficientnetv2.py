
import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if expansion != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_residual:
            out += identity
        return out

class EfficientNetV2L(nn.Module):
    def __init__(self, num_classes=3):
        super(EfficientNetV2L, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 32, expansion=1, stride=1),
            MBConvBlock(32, 64, expansion=4, stride=2),
            MBConvBlock(64, 128, expansion=4, stride=2),
            MBConvBlock(128, 160, expansion=6, stride=2),
            MBConvBlock(160, 256, expansion=6, stride=1),
            MBConvBlock(256, 320, expansion=6, stride=2),
            MBConvBlock(320, 640, expansion=6, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(640, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def load_pretrained(self, weight_path):
        state_dict = torch.load(weight_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
