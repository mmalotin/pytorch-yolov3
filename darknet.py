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


'''
model = Darknet([32, 64, 128, 256, 512], [1, 2, 8, 8, 4])
st_d = torch.load('dn53_weights.pth')
model.load_state_dict(st_d)
im = Image.open('/home/nick/D/study/Python/HackerRank/fnst/images/dancing.jpg')
tfms = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
x = tfms(im).unsqueeze(0)
res = model(x)
'''


'''
def flatten_model(model, lst):
    if list(model.children()) == []:
        lst.append(model)
    for ch in model.children():
        flatten_model(ch, lst)

flat_model = []
flatten_model(model, flat_model)
flat_model

with open('/home/nick/D/study/Python/HackerRank/yolov3/yolov3.weights', 'rb') as f:
    weights = np.fromfile(f, dtype=np.float32)

weights.shape
ptr = 0
bn, cv = flat_model[1], flat_model[0]


def conv_layer_weights(bn, conv):
    global ptr
    global weights
    l = bn.bias.numel()
    bn_bias = torch.from_numpy(weights[ptr:(ptr+l)]).view_as(bn.bias)
    ptr += l
    l = bn.weight.numel()
    bn_w = torch.from_numpy(weights[ptr:(ptr+l)]).view_as(bn.weight)
    ptr += l
    l = bn.running_mean.numel()
    bn_rm = torch.from_numpy(weights[ptr:(ptr+l)]).view_as(bn.running_mean)
    ptr += l
    l = bn.running_var.numel()
    bn_rv = torch.from_numpy(weights[ptr:(ptr+l)]).view_as(bn.running_var)
    ptr += l
    bn.bias.data.copy_(bn_bias)
    bn.weight.data.copy_(bn_w)
    bn.running_mean.data.copy_(bn_rm)
    bn.running_var.data.copy_(bn_rv)

    l = conv.weight.numel()
    conv_w = torch.from_numpy(weights[ptr:(ptr+l)]).view_as(conv.weight)
    ptr += l
    conv.weight.data.copy_(conv_w)

conv_layer_weights(bn, cv)

bn.bias
model.extractor[0].bn.bias
len(flat_model)

bns = flat_model[:-1][1::2]
convs = flat_model[:-1][::2]

for bn, conv in zip(bns, convs):
    conv_layer_weights(bn, conv)


torch.save(model.state_dict(), 'dn53_weights.pth')
'''
