import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CoConv']

class route_func(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=3, reduction=16, activation='sigmoid'):
        super().__init__()

        self.activation = activation
        self.num_experts = num_experts
        self.out_channels = out_channels

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
        
        if self.activation == 'sigmoid':
            self.activation_func = nn.Sigmoid()
        else:
            # self.temperature = 30
            self.activation_func = nn.Softmax(1)

    def forward(self, x):
        b, _, _, _ = x.size()
        a1 = self.gap1(x)
        a3 = self.gap3(x)
        a1 = a1.expand_as(a3)
        attention = torch.cat([a1, a3], dim=1)
        if self.activation == 'sigmoid':
            attention = self.activation_func(self.dwise_separable(attention))
        else:
            attention = self.dwise_separable(attention).view(b, self.num_experts, self.out_channels)
            attention = self.activation_func(attention).view(b, -1)
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
    def __init__(self, in_channels, out_channels, kernel_size, num_experts=3, stride=1, padding=0, groups=1, reduction=16, bias=False, fuse_conv=False, activation='sigmoid'):
        super().__init__()
        self.fuse_conv = fuse_conv
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.groups = groups

        # routing function
        self.routing_func = route_func(in_channels, out_channels, num_experts, reduction, activation)
        
        if fuse_conv:
            self.kernel_size = kernel_size
            self.convs = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels, kernel_size, kernel_size)) # to count parameters during inference

            if bias:
                self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
            else:
                self.register_parameter('bias', None)
            self.bns = nn.BatchNorm2d(out_channels)
        else:
            self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias) for i in range(num_experts)])
            self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(num_experts)])

    def forward(self, x):
        routing_weight = self.routing_func(x) # N x k*C
        if self.fuse_conv:
            routing_weight = routing_weight.view(-1, self.num_experts, self.out_channels).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            b, c_in, h, w = x.size()
            x = x.view(1, -1, h, w)
            weight = self.convs.unsqueeze(0)
            combined_weight = (weight * routing_weight).view(self.num_experts, b*self.out_channels, c_in, self.kernel_size, self.kernel_size)
            combined_weight = torch.sum(combined_weight, dim=0)
            if self.bias is not None:
                combined_bias = routing_weight.squeeze(-1).squeeze(-1).squeeze(-1).view(-1, self.num_experts * self.out_channels) * self.bias.view(-1).unsqueeze(0)
                combined_bias = combined_bias.sum(1)
                output = F.conv2d(x, weight=combined_weight, 
                                stride=self.stride, padding=self.padding, groups=self.groups * b)
            else:
                output = F.conv2d(x, weight=combined_weight,
                                stride=self.stride, padding=self.padding, groups=self.groups * b)
            output = self.bns(output.view(b, self.out_channels, output.size(-2), output.size(-1)))
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
    conv = CoConv(16, 64, 3, padding=1, fuse_conv=False)
    y = conv(x)
    print(y.shape)
    conv = CoConv(16, 64, 3, padding=1, fuse_conv=True)
    y = conv(x)
    print(y.shape)
    conv = CoConv(16, 64, 3, padding=1, fuse_conv=True, bias=True)
    y = conv(x)
    print(y.shape)
    conv = CoConv(16, 64, 3, padding=1, fuse_conv=True, bias=True, activation='softmax')
    y = conv(x)
    print(y.shape)

test()