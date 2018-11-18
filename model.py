import torch
import torch.nn as nn
import torch.nn.functional as F

def padding(f):
    return (f - 1) // 2

def Conv2dSame(in_channels, out_channels, kernel_size, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding(kernel_size))

class ConvolutionalNeuralNet(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Layers
        self.conv2d1 = Conv2dSame(3, 32, 5, stride=2) # (3, 128, 256) -> (32, 64, 128)
        self.conv2d2 = Conv2dSame(32, 32, 5, stride=2) # -> (32, 32, 64)
        self.conv2d3 = Conv2dSame(32, 32, 5, stride=2) # -> (32, 16, 34)
        self.conv2d4 = Conv2dSame(32, 32, 5, stride=2) # -> (32, 8, 16)
        self.conv2d5 = Conv2dSame(32, 64, 5, stride=2) # -> (64, 4, 8)
        self.linear1 = nn.Linear(2048, 1307)

    def forward(self, x):
        conv_1 = F.elu(self.conv2d1(x))
        conv_2 = F.elu(self.conv2d2(conv_1))
        conv_3 = F.elu(self.conv2d3(conv_2))
        conv_4 = F.elu(self.conv2d4(conv_3))
        conv_5 = F.elu(self.conv2d5(conv_4))
        conv_5_reshape = conv_5.view(-1, 2048)
        linear_1 = F.elu(self.linear1(conv_a5_reshape))
        linear_a2 = nn.Softmax(self.linear2(linear_a1))
        return linear_a2