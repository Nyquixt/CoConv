import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['CoConv']

class route_func(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=3, reduction=16, activation='sigmoid'):
        super().__init__()
        # Global Average Pool
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap3 = nn.AdaptiveAvgPool2d(3)

        squeeze_channels = max(in_channels // reduction, reduction)
        
        self.dwise_separable = nn.Sequential(
            nn.Conv2d(2 * in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=3, stride=1, groups=squeeze_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, num_experts * out_channels, kernel_size=1, stride=1, groups=1, bias=False)
        )
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            # self.temperature = 30
            self.activation = nn.Softmax(1)

    def forward(self, x):
        b, _, _, _ = x.size()
        a1 = self.gap1(x)
        a3 = self.gap3(x)
        a1 = a1.expand_as(a3)
        attention = torch.cat([a1, a3], dim=1)
        attention = self.activation(self.dwise_separable(attention))
        return attention

class route_func_single_scale(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=3, reduction=16):
        super().__init__()
        # Global Average Pool
        self.gap1 = nn.AdaptiveAvgPool2d(1)

        squeeze_channels = max(in_channels // reduction, reduction)
        
        self.dwise_separable = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=1, stride=1, groups=squeeze_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, num_experts * out_channels, kernel_size=1, stride=1, groups=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, _, _ = x.size()
        a1 = self.gap1(x)
        attention = self.sigmoid(self.dwise_separable(a1))
        return attention

class CoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_experts=3, stride=1, padding=0, groups=1, reduction=16, bias=False, deploy=False, activation='sigmoid'):
        super().__init__()
        self.deploy = deploy
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        self.padding = padding
        self.groups = groups

        # routing function
        self.routing_func = route_func(in_channels, out_channels, num_experts, reduction, activation)

        # convs
        if deploy: # for the purpose of testing inference time only. loading state_dict is not implemented
            self.convs = [nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)) for i in range(num_experts)]
            if bias:
                self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
            else:
                self.register_parameter('bias', None)
        else:
            self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias) for i in range(num_experts)])
            self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(num_experts)])

    def forward(self, x):
        routing_weight = self.routing_func(x) # N x k*C
        if self.deploy:
            routing_weight = self.routing_func(x) # N x k*C
            combined_weight = torch.zeros(*self.convs[0].size())
            for i in range(self.num_experts):
                route = torch.sum(routing_weight[:, i * self.out_channels : (i+1) * self.out_channels], dim=0)
                combined_weight += self.convs[i] * route.unsqueeze(-1).expand_as(self.convs[i])
            
            if self.bias is not None:
                combined_bias = routing_weight.squeeze(-1).squeeze(-1).squeeze(-1).view(-1, self.num_experts * self.out_channels) * self.bias.view(-1).unsqueeze(0)
                combined_bias = combined_bias.sum(1)
                output = F.conv2d(x, weight=combined_weight, 
                                stride=self.stride, padding=self.padding, groups=self.groups)
            else:
                output = F.conv2d(x, weight=combined_weight,
                                stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            outputs = []
            for i in range(self.num_experts):
                route = routing_weight[:, i * self.out_channels : (i+1) * self.out_channels]
                # X * W
                out = self.convs[i](x)
                out = self.bns[i](out)
                out = out * route.expand_as(out)
                outputs.append(out)
            output = sum(outputs)
        return output

def test():
    x = torch.randn(64, 16, 32, 32)
    conv = CoConv(16, 64, 3, padding=1, deploy=False)
    y = conv(x)
    print(y.shape)
    conv = CoConv(16, 64, 3, padding=1, deploy=True)
    y = conv(x)
    print(y.shape)
    conv = CoConv(16, 64, 3, padding=1, deploy=True, bias=True)
    y = conv(x)
    print(y.shape)

# test()