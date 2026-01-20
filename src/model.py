'''import torch, torch.nn as nn, utilities

class ApplicationStack(nn.Module):
    # stack of Conv -> BN -> ReLU because conv1(...), bn1(...), etc. feels clumsy
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1,
                              stride=stride,
                              bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FacialEmotionRecognitionCNN(nn.Module):
    # in- and out-channels are inferred from list_channels; usage self-explanatory
    # strides inferred from out-channels, downsampling when entering new stage (except first)

    def __init__(self, 
                 list_channels = [3, 32, 32, 64, 64, 128, 128, 256, 256], 
                 n_classes: int = 6):
        
        
        super().__init__()
        self.list_in_channels = list_channels[:-1]
        self.list_out_channels = list_channels[1:]

        # with default argument the code would infer [1, 1, 2, 1, 2, 1, 2, 1] for strides
        seen = set(); seen.add(self.list_out_channels[0])
        self.list_strides = [1 if c in seen else (seen.add(c) or 2)
                             for c in self.list_out_channels]
        del seen

        self.stacks = nn.ModuleList([ ApplicationStack(in_channels=in_channels,
                                         out_channels=out_channels,
                                         stride=stride)   
                        for in_channels, out_channels, stride in 
                        zip(self.list_in_channels,
                            self.list_out_channels,
                            self.list_strides
                    )])
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.list_out_channels[-1], n_classes)

        self.timing_supervisor = [0.0, 0.0]
    
    def reset_timing_supervisor(self):
        self.timing_supervisor = [0.0, 0.0]
    
    def forward(self, x):
        with utilities.Timer("MODEL forward", show=False) as t:
            for stack in self.stacks: x = stack(x)
            x = self.gap(x); x = torch.flatten(x, 1); x = self.fc(x)
        
        self.timing_supervisor[0] += t.timer
        self.timing_supervisor[1] += 1
        return x '''''

import torch
import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # eski training loop buna bakıyor
        self.timing_supervisor = [0.0, 0]

    def forward(self, x):
        return self.model(x)

    def reset_timing_supervisor(self):
        self.timing_supervisor = [0.0, 0]
