# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd
import jittor as jt
from jittor import nn

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
        self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = nn.relu(x)
        channel_attention = jt.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)  # 输入通道数维度
        filter_attention = jt.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)  # 输出通道数维度
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)  # 卷积核数量维度
        kernel_attention = nn.softmax(kernel_attention / self.temperature, dim=1)
        return channel_attention, filter_attention, kernel_attention


class MCA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, no_res=False,
                 reduction=0.0625, kernel_num=4):
        super(MCA, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.no_res = no_res
        # if not self.no_res:
        #     self.conv1 = nn.Conv2d(in_planes, 256, 1, 1, 0)

        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(jt.randn(kernel_num, out_planes, in_planes, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        print('#######MCA########')

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def execute(self, x):
        # shortcut = x  # add
        # if not self.no_res:
        #     shortcut = self.conv1(x)
        channel_attention, filter_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = jt.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes, self.kernel_size, self.kernel_size])
        output = nn.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        # if not self.no_r1`rtcut + output  # add

        return output


if __name__ == '__main__':

    odconv = MCA(128, 256, kernel_size=3, stride=1, padding=1)

    x = jt.randn(2, 128, 104, 136)

    out = odconv(x)
    print(out.shape)
