import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(4):
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class ConjUNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)

        self.up3 = UpConv(512, 256)
        self.up2 = UpConv(256, 128)
        self.up1 = UpConv(128, 64)
        self.up_conv3 = ConvBlock(512, 256)
        self.up_conv2 = ConvBlock(256, 128)
        self.up_conv1 = ConvBlock(128, 64)

        self.out_a = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.out_creal = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.out_cimag = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.out_creal.bias, 0.1)
        nn.init.constant_(self.out_cimag.bias, 0.1)

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool(e1))
        e3 = self.conv3(self.pool(e2))
        e4 = self.conv4(self.pool(e3))

        d4 = self.up_conv3(torch.cat((e3, self.up3(e4)), dim=1))
        d3 = self.up_conv2(torch.cat((e2, self.up2(d4)), dim=1))
        feat = self.up_conv1(torch.cat((e1, self.up1(d3)), dim=1))

        a = torch.sigmoid(self.out_a(feat))
        C_real = self.out_creal(feat)
        C_imag = self.out_cimag(feat)
        return a, C_real, C_imag
