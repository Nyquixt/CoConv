import torch
import torch.nn as nn
import torch.nn.functional as F
from convs.coconv import *

__all__ = ['CoConv_ResNet18']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, num_experts=3, fuse_conv=False, activation='sigmoid'):
        super().__init__()
        self.conv1 = CoConv(in_channels, channels, kernel_size=3, stride=stride, padding=1, num_experts=num_experts, fuse_conv=fuse_conv, activation=activation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = CoConv(channels, channels, kernel_size=3, stride=1, padding=1, num_experts=num_experts, fuse_conv=fuse_conv, activation=activation)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Addition
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CoConv_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, num_experts=3, fuse_conv=False, activation='sigmoid'):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, num_experts=num_experts, fuse_conv=fuse_conv, activation=activation)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, num_experts=num_experts, fuse_conv=fuse_conv, activation=activation)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, num_experts=num_experts, fuse_conv=fuse_conv, activation=activation)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, num_experts=num_experts, fuse_conv=fuse_conv, activation=activation)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride, num_experts, fuse_conv, activation):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride, num_experts, fuse_conv, activation))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def CoConv_ResNet18(num_experts=3, fuse_conv=False, activation='sigmoid'):
    return CoConv_ResNet(BasicBlock, [2, 2, 2, 2], num_experts=num_experts, fuse_conv=fuse_conv, activation=activation)