import torch
import torch.nn as nn
import torch.nn.functional as F

def padding(f):
    return (f - 1) // 2

def Conv2dSame(in_channels, out_channels, kernel_size, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding(kernel_size))

class MultiClassifier(nn.Module):
    def __init__(self):
        super(MultiClassifier, self).__init__()
        # Layers
        self.conv2d1 = Conv2dSame(3, 32, 5, stride=2) # (3, 128, 256) -> (32, 64, 128)
        self.conv2d2 = Conv2dSame(32, 64, 5, stride=2) # -> (64, 32, 64)
        self.conv2d3 = Conv2dSame(64, 128, 5, stride=2) # -> (128, 16, 34)
        self.conv2d4 = Conv2dSame(128, 256, 5, stride=2) # -> (256, 8, 16)
        self.conv2d5 = Conv2dSame(256, 512, 5, stride=2) # -> (512, 4, 8)
        #self.linear1 = nn.Linear(2048, 1024)
        #self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(16384, 28)

    def forward(self, x):
        conv_1 = F.elu(self.conv2d1(x))
        conv_2 = F.elu(self.conv2d2(conv_1))
        conv_3 = F.elu(self.conv2d3(conv_2))
        conv_4 = F.elu(self.conv2d4(conv_3))
        conv_5 = F.elu(self.conv2d5(conv_4))
        conv_5_reshape = conv_5.view(-1, 16384)
        linear_1 = self.linear3(conv_5_reshape)
        #linear_2 = F.elu(self.linear2(linear_1))
        #linear_3 = F.elu(self.linear3(linear_2))
        #final_linear = F.softmax(linear_3)
        return linear_1

class MultiLabelClassifier(nn.Module):
    def __init__(self):
        super(MultiLabelClassifier, self).__init__()
        # Layers
        self.conv2d1 = Conv2dSame(3, 32, 5, stride=2) # (3, 128, 256) -> (32, 64, 128)
        self.conv2d2 = Conv2dSame(32, 64, 5, stride=2) # -> (64, 32, 64)
        self.conv2d3 = Conv2dSame(64, 128, 5, stride=2) # -> (128, 16, 32)
        self.conv2d4 = Conv2dSame(128, 256, 5, stride=2) # -> (256, 8, 16)
        self.conv2d5 = Conv2dSame(256, 512, 5, stride=2) # -> (512, 4, 8)
        #self.linear1 = nn.Linear(2048, 1024)
        #self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(16384, 28)

    def forward(self, x):
        conv_1 = F.elu(self.conv2d1(x))
        conv_2 = F.elu(self.conv2d2(conv_1))
        conv_3 = F.elu(self.conv2d3(conv_2))
        conv_4 = F.elu(self.conv2d4(conv_3))
        conv_5 = F.elu(self.conv2d5(conv_4))
        conv_5_reshape = conv_5.view(-1, 16384)
        linear_1 = self.linear3(conv_5_reshape)
        #linear_2 = F.elu(self.linear2(linear_1))
        #linear_3 = F.elu(self.linear3(linear_2))
        #final_linear = F.sigmoid(linear_3)
        return linear_1
