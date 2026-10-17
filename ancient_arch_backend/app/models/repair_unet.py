import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class RepairUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)   # 改这里

        self.up1 = Up(1024, 512)       # 改这里
        self.up2 = Up(512, 256)        # 改这里
        self.up3 = Up(256, 128)        # 改这里
        self.up4 = Up(128, 64)         # 改这里

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)     # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 1024

        x = self.up1(x5, x4) # 512
        x = self.up2(x, x3)  # 256
        x = self.up3(x, x2)  # 128
        x = self.up4(x, x1)  # 64

        x = self.outc(x)
        x = self.act(x)
        return x