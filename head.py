import torch
import torch.nn as nn
import torch.nn.functional as F
from darknet import Darknet
from route_ext import RouteExtractor
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np


class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(c_out, momentum=0.01)

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.act:
            return F.leaky_relu(out, negative_slope=0.1)
        else:
            return out


class DetectionBlock(nn.Module):
    def __init__(self, c_in, c_inter, c_out, route=True):
        super().__init__()
        self.route = route
        self.conv1 = BasicConv(c_in, c_inter, 1)
        self.conv2 = BasicConv(c_inter, c_inter*2, 3, pad=3//2)
        self.conv3 = BasicConv(c_inter*2, c_inter, 1)
        self.conv4 = BasicConv(c_inter, c_inter*2, 3, pad=3//2)
        self.conv5 = BasicConv(c_inter*2, c_inter, 1)
        self.conv6 = BasicConv(c_inter, c_inter*2, 3, pad=3//2)
        self.conv7 = nn.Conv2d(c_inter*2, c_out, 1, 1)

    def forward(self, x):
        route = self.conv1(x)
        route = self.conv2(route)
        route = self.conv3(route)
        route = self.conv4(route)
        route = self.conv5(route)
        out = self.conv6(route)
        out = self.conv7(out)
        if self.route:
            return route, out
        else:
            return out


class UpsampleMerge(nn.Module):
    def __init__(self, c_in, c_inter):
        super().__init__()
        self.conv = BasicConv(c_in , c_inter, 1)

    def forward(self, route_head, route_backbone):
        out = F.upsample(self.conv(route_head), scale_factor=2)
        out = torch.cat([out, route_backbone], 1)
        return out


class Yolo3(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone_routes = RouteExtractor(self.backbone, [3, 4])
        self.detector1 = DetectionBlock(1024, 512, 255)
        self.upmerger1 = UpsampleMerge(512, 256)
        self.detector2 = DetectionBlock(768, 256, 255)
        self.upmerger2 = UpsampleMerge(256, 128)
        self.detector3 = DetectionBlock(384, 128, 255, route=False)

    def forward(self, x):
        out = self.backbone(x)
        route1, out1 = self.detector1(out)
        out = self.upmerger1(route1, self.backbone_routes.routes[1])
        route2, out2 = self.detector2(out)
        out = self.upmerger2(route2, self.backbone_routes.routes[0])
        out3 = self.detector3(out)
        return [out1, out2, out3]
