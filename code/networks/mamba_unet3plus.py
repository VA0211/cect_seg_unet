import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.mamba_vision import mamba_vision_tiny

# ========== Decoder Block ==========
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)]
        if act:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

# ========== UNet 3+ with Mamba Backbone ==========
class UNet3PlusMamba(nn.Module):
    def __init__(self, num_classes=1, input_channels=1):
        super().__init__()
        # Use pretrained MambaVision tiny model
        self.backbone = mamba_vision_tiny(pretrained=True, features_only=True, in_chans=input_channels)
        # Returns [stage1, stage2, stage3, stage4]
        # Channel dims: tiny = [64, 128, 320, 512]
        chs = [64, 128, 320, 512]
        out_ch = 64

        # Decoder4 (lowest scale)
        self.conv_e1_d4 = ConvBlock(chs[0], out_ch)
        self.conv_e2_d4 = ConvBlock(chs[1], out_ch)
        self.conv_e3_d4 = ConvBlock(chs[2], out_ch)
        self.conv_e4_d4 = ConvBlock(chs[3], out_ch)
        self.finalblock_d4 = ConvBlock(out_ch * 4, out_ch * 4)

        # Decoder3
        self.conv_e1_d3 = ConvBlock(chs[0], out_ch)
        self.conv_e2_d3 = ConvBlock(chs[1], out_ch)
        self.conv_e3_d3 = ConvBlock(chs[2], out_ch)
        self.conv_d4_d3 = ConvBlock(out_ch * 4, out_ch)
        self.finalblock_d3 = ConvBlock(out_ch * 4, out_ch * 4)

        # Decoder2
        self.conv_e1_d2 = ConvBlock(chs[0], out_ch)
        self.conv_e2_d2 = ConvBlock(chs[1], out_ch)
        self.conv_d3_d2 = ConvBlock(out_ch * 4, out_ch)
        self.conv_d4_d2 = ConvBlock(out_ch * 4, out_ch)
        self.finalblock_d2 = ConvBlock(out_ch * 4, out_ch * 4)

        # Decoder1
        self.conv_e1_d1 = ConvBlock(chs[0], out_ch)
        self.conv_d2_d1 = ConvBlock(out_ch * 4, out_ch)
        self.conv_d3_d1 = ConvBlock(out_ch * 4, out_ch)
        self.conv_d4_d1 = ConvBlock(out_ch * 4, out_ch)
        self.finalblock_d1 = ConvBlock(out_ch * 4, out_ch * 4)

        # Output
        self.out_conv = nn.Conv2d(out_ch * 4, num_classes, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        # Backbone features: [C1, C2, C3, C4]
        feats = self.backbone(x)
        e1, e2, e3, e4 = feats

        # Decoder 4
        e1_d4 = self.conv_e1_d4(F.max_pool2d(e1, 8))
        e2_d4 = self.conv_e2_d4(F.max_pool2d(e2, 4))
        e3_d4 = self.conv_e3_d4(F.max_pool2d(e3, 2))
        e4_d4 = self.conv_e4_d4(e4)
        d4 = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4], dim=1)
        d4 = self.finalblock_d4(d4)

        # Decoder 3
        e1_d3 = self.conv_e1_d3(F.max_pool2d(e1, 4))
        e2_d3 = self.conv_e2_d3(F.max_pool2d(e2, 2))
        e3_d3 = self.conv_e3_d3(e3)
        d4_d3 = self.conv_d4_d3(F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True))
        d3 = torch.cat([e1_d3, e2_d3, e3_d3, d4_d3], dim=1)
        d3 = self.finalblock_d3(d3)

        # Decoder 2
        e1_d2 = self.conv_e1_d2(F.max_pool2d(e1, 2))
        e2_d2 = self.conv_e2_d2(e2)
        d3_d2 = self.conv_d3_d2(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True))
        d4_d2 = self.conv_d4_d2(F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=True))
        d2 = torch.cat([e1_d2, e2_d2, d3_d2, d4_d2], dim=1)
        d2 = self.finalblock_d2(d2)

        # Decoder 1
        e1_d1 = self.conv_e1_d1(e1)
        d2_d1 = self.conv_d2_d1(F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True))
        d3_d1 = self.conv_d3_d1(F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True))
        d4_d1 = self.conv_d4_d1(F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True))
        d1 = torch.cat([e1_d1, d2_d1, d3_d1, d4_d1], dim=1)
        d1 = self.finalblock_d1(d1)

        out = self.out_conv(d1)
        return self.out_act(out)
