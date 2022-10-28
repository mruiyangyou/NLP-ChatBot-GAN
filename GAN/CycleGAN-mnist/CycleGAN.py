import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv(c_in, c_out, k_size, stride = 2, pad = 1, bn = True, return_li = False):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return layers if return_li else nn.Sequential(*layers)


def deconv(c_in, c_out, k_size, stride = 2, pad = 1, bn = True, return_li = False):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return layers if return_li else nn.Sequential(*layers)


class G(nn.Module):
    def __init__(self, input_dim, output_dim, conv_dim = 64):
        super(G, self).__init__()
        self.conv1 = Conv(input_dim, conv_dim, 4)
        self.conv2 = Conv(conv_dim, conv_dim*2, 4)

        self.conv3 = Conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = Conv(conv_dim*2, conv_dim*2, 3, 1, 1)

        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, output_dim, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)

        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)

        out = F.leaky_relu(self.deconv1(out), 0.05)
        out = self.deconv2(out)
        return F.tanh(out)

class D(nn.Module):
    def __init__(self,input_dim, conv_dim, num_layer):
        super(D, self).__init__()
        sequence = []
        sequence += Conv(input_dim, conv_dim, 4, bn = False, return_li=True)
        sequence += [nn.LeakyReLU(0.05)]
        for i in range(num_layer-1):
            a, b = 2 ** i, 2 **(i+1)
            sequence += Conv(conv_dim * a, conv_dim * b, 4, return_li=True)
            sequence += [nn.LeakyReLU(0.05)]

        sequence += Conv(conv_dim * (2 ** (num_layer-1)), 1, 4, 1, 0, bn=False, return_li=True)
        self.convs = nn.Sequential(*sequence)

    def forward(self, x):
        return self.convs(x)














