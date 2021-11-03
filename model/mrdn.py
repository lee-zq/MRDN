from model import common
from model import block as B

import torch.nn as nn
import torch


def make_model(args, parent=False):
    return MRDN(args)


class EPA(nn.Module):
    '''EPA is efficient pixel attention'''

    def __init__(self, channel):
        super(EPA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, 1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        out = torch.mul(x, y)
        return out


class gconv(nn.Module):
    def __init__(self, channel):
        super(gconv, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = y + x
        y = self.relu(y)
        return y


class MRDB(nn.Module):
    def __init__(self, channel):
        super(MRDB, self).__init__()

        self.channel = channel
        self.conv = nn.Conv2d(channel, channel, kernel_size=1, padding=0)

        self.body1 = nn.Sequential(gconv(channel))
        self.body2 = nn.Sequential(gconv(channel * 3 // 4))
        self.body3 = nn.Sequential(gconv(channel * 2 // 4))
        self.body4 = nn.Sequential(gconv(channel * 1 // 4))

        self.CA1 = EPA(channel * 1 // 4)
        self.CA2 = EPA(channel * 1 // 4)
        self.CA3 = EPA(channel * 1 // 4)

        self.CA = B.ESA(channel, nn.Conv2d)

    def forward(self, x):
        res = self.body1(x)
        res, g1 = torch.split(res, (self.channel * 3 // 4, self.channel * 1 // 4), dim=1)
        res = self.body2(res)
        res, g2 = torch.split(res, (self.channel * 2 // 4, self.channel * 1 // 4), dim=1)
        res = self.body3(res)
        res, g3 = torch.split(res, (self.channel * 1 // 4, self.channel * 1 // 4), dim=1)
        g4 = self.body4(res)

        y = torch.cat([self.CA1(g1), self.CA2(g2), self.CA3(g3), g4], dim=1)
        # y = torch.cat([g1, g2, g3, g4], dim=1)
        y = self.conv(y)
        y = self.CA(y)

        return y + x


class MRDN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MRDN, self).__init__()

        n_feats = args.n_feats  # 192
        kernel_size = 3
        scale = args.scale[0]  # 4
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)  # 标准化
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.n_feats = n_feats

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]  # [3,64,3]

        # define tail module
        m_tail = [
            B.pixelshuffle_block(n_feats, 3, upscale_factor=4)
        ]

        self.tail = nn.Sequential(*m_tail)
        self.head = nn.Sequential(*m_head)

        self.conv1 = nn.Conv2d(8 * n_feats, n_feats, kernel_size=1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        self.B1 = MRDB(n_feats)
        self.B2 = MRDB(n_feats)
        self.B3 = MRDB(n_feats)
        self.B4 = MRDB(n_feats)
        self.B5 = MRDB(n_feats)
        self.B6 = MRDB(n_feats)
        self.B7 = MRDB(n_feats)
        self.B8 = MRDB(n_feats)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        out_B1 = self.B1(x)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)
        y = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        y = self.conv2(self.conv1(y))

        y = y + x  # work

        y = self.tail(y)

        y = self.add_mean(y)
        return y


import option
from torchsummaryX import summary

if __name__ == '__main__':
    model = MRDN(option.args)
    print(model)
    in1 = torch.randn(1, 3, 64, 64)
    out = model(in1)
    # summary(model, input_size=(3, 224, 224), batch_size=-1, device='cpu')
    summary(model, torch.zeros((1, 3, 320, 180)))
    print(out.size())
