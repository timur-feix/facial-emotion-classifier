import torch
import torch.nn as nn


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (sadece boyut değişirse)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out

class LightResNet18(nn.Module):
    def __init__(self, num_classes=6, widths=(32, 64, 128, 256)):
        super().__init__()
        w1, w2, w3, w4 = widths

        self.stem = nn.Sequential(
            nn.Conv2d(3, w1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=False),
        )

        self.layer1 = self._make_layer(w1, w1, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(w1, w2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(w2, w3, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(w3, w4, num_blocks=2, stride=2)

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(w4, num_classes)

    def _make_layer(self, in_c, out_c, num_blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x) 
        return self.fc(x)


