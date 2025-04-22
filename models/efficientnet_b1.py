import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        reduced_channels = int(in_channels * se_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, se_ratio=0.25):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ]

        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            SqueezeExcite(hidden_dim, se_ratio),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out

class EfficientNetB1(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # Format: (in_c, out_c, exp, stride, repeats)
        blocks_args = [
            (32, 16, 1, 1, 1),
            (16, 24, 6, 2, 2),
            (24, 40, 6, 2, 2),
            (40, 80, 6, 2, 3),
            (80, 112, 6, 1, 3),
            (112, 192, 6, 2, 4),
            (192, 320, 6, 1, 1)
        ]

        layers = []
        for in_c, out_c, exp, stride, reps in blocks_args:
            layers.append(MBConv(in_c, out_c, exp, stride))
            for _ in range(reps - 1):
                layers.append(MBConv(out_c, out_c, exp, 1))
        self.blocks = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

    def load_pretrained(self, weight_path):
        state_dict = torch.load(weight_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
