import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride=1, pad=0):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(c_out, momentum=0.01)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)


class ResidualBlock(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv_reduce = BasicConv(c_in, c_in//2, 1)
        self.conv_expand = BasicConv(c_in//2, c_in, 3, pad=3//2)

    def forward(self, x):
        out = self.conv_reduce(x)
        out = self.conv_expand(out)
        return x + out


class DarknetBlock(nn.Module):
    def __init__(self, c_in, length):
        super().__init__()
        self.conv_down = BasicConv(c_in, c_in*2, 3, 2, 3//2)
        self.res_blocks = nn.Sequential(*[ResidualBlock(c_in*2)
                                          for i in range(length)])

    def forward(self, x):
        out = self.conv_down(x)
        out = self.res_blocks(out)
        return out


class Darknet(nn.Module):
    def __init__(self, in_sizes=[32, 64, 128, 256, 512], lens=[1, 2, 8, 8, 4]):
        super().__init__()
        conv1 = BasicConv(3, 32, 3, pad=3//2)
        dn_blocks = [DarknetBlock(in_size, l)
                     for in_size, l in zip(in_sizes, lens)]
        self.extractor = nn.Sequential(*([conv1] + dn_blocks))
        self.fc = nn.Linear(in_sizes[-1]*2, 1000)

    def forward(self, x):
        out = self.extractor(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
