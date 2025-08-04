import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if act else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UNet3plus(nn.Module):
    def __init__(self, img_size=(256,256,1), num_classes=2, base_filters=64, pretrained=True, in_ch=1):
        super().__init__()
        # --- Pretrained ResNet50 Encoder ---
        resnet = resnet50(pretrained=pretrained)
        if in_ch != 3:
            # update first conv to support n-channel input
            weight = resnet.conv1.weight.clone()
            resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if in_ch == 1:
                resnet.conv1.weight.data = weight.sum(dim=1, keepdim=True)
            else:
                resnet.conv1.weight.data[:, :3] = weight
        # Encoder layers
        self.enc0 = nn.Identity() # will be input x itself, for compatibility
        self.enc1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.enc2 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        )
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        # standardize output channels
        f = base_filters
        self.stem_conv = ConvBlock(1 if in_ch == 1 else in_ch, f)
        self.d_conv = nn.ModuleList([ConvBlock(c, f) for c in [64, 256, 512, 1024, 2048]])
        # each decoder fusion, conv to 64 x H x W
        self.conv_e1_d4 = ConvBlock(f, f); self.conv_e2_d4 = ConvBlock(f, f)
        self.conv_e3_d4 = ConvBlock(f, f); self.conv_e4_d4 = ConvBlock(f, f)
        self.conv_e5_d4 = ConvBlock(f, f); self.conv_d4 = ConvBlock(f*5, f*5)
        self.conv_e1_d3 = ConvBlock(f, f); self.conv_e2_d3 = ConvBlock(f, f)
        self.conv_e3_d3 = ConvBlock(f, f); self.conv_d4_d3 = ConvBlock(f, f)
        self.conv_e5_d3 = ConvBlock(f, f); self.conv_d3 = ConvBlock(f*5, f*5)
        self.conv_e1_d2 = ConvBlock(f, f); self.conv_e2_d2 = ConvBlock(f, f)
        self.conv_d3_d2 = ConvBlock(f, f); self.conv_d4_d2 = ConvBlock(f, f)
        self.conv_e5_d2 = ConvBlock(f, f); self.conv_d2 = ConvBlock(f*5, f*5)
        self.conv_e1_d1 = ConvBlock(f, f); self.conv_d2_d1 = ConvBlock(f, f)
        self.conv_d3_d1 = ConvBlock(f, f); self.conv_d4_d1 = ConvBlock(f, f)
        self.conv_e5_d1 = ConvBlock(f, f); self.conv_d1 = ConvBlock(f*5, f*5)
        self.out_conv = nn.Conv2d(f*5, num_classes, 3, padding=1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        # All encoder outputs to 64 channels
        e1_s = self.d_conv[0](F.interpolate(e1, size=x.shape[2:], mode='bilinear', align_corners=False))
        e2_s = self.d_conv[1](F.interpolate(e2, size=x.shape[2:], mode='bilinear', align_corners=False))
        e3_s = self.d_conv[2](F.interpolate(e3, size=x.shape[2:], mode='bilinear', align_corners=False))
        e4_s = self.d_conv[3](F.interpolate(e4, size=x.shape[2:], mode='bilinear', align_corners=False))
        e5_s = self.d_conv[4](F.interpolate(e5, size=x.shape[2:], mode='bilinear', align_corners=False))

        # Decoder 4
        e1_d4 = self.conv_e1_d4(F.max_pool2d(e1_s, 8))
        e2_d4 = self.conv_e2_d4(F.max_pool2d(e2_s, 4))
        e3_d4 = self.conv_e3_d4(F.max_pool2d(e3_s, 2))
        e4_d4 = self.conv_e4_d4(e4_s)
        e5_d4 = self.conv_e5_d4(F.interpolate(e5_s, size=e4_s.shape[2:], mode='bilinear', align_corners=False))
        d4 = self.conv_d4(torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], dim=1))

        # Decoder 3
        e1_d3 = self.conv_e1_d3(F.max_pool2d(e1_s, 4))
        e2_d3 = self.conv_e2_d3(F.max_pool2d(e2_s, 2))
        e3_d3 = self.conv_e3_d3(e3_s)
        d4_d3 = self.conv_d4_d3(F.interpolate(d4, size=e3_s.shape[2:], mode='bilinear', align_corners=False))
        e5_d3 = self.conv_e5_d3(F.interpolate(e5_s, size=e3_s.shape[2:], mode='bilinear', align_corners=False))
        d3 = self.conv_d3(torch.cat([e1_d3, e2_d3, e3_d3, d4_d3, e5_d3], dim=1))

        # Decoder 2
        e1_d2 = self.conv_e1_d2(F.max_pool2d(e1_s, 2))
        e2_d2 = self.conv_e2_d2(e2_s)
        d3_d2 = self.conv_d3_d2(F.interpolate(d3, size=e2_s.shape[2:], mode='bilinear', align_corners=False))
        d4_d2 = self.conv_d4_d2(F.interpolate(d4, size=e2_s.shape[2:], mode='bilinear', align_corners=False))
        e5_d2 = self.conv_e5_d2(F.interpolate(e5_s, size=e2_s.shape[2:], mode='bilinear', align_corners=False))
        d2 = self.conv_d2(torch.cat([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2], dim=1))

        # Decoder 1
        e1_d1 = self.conv_e1_d1(e1_s)
        d2_d1 = self.conv_d2_d1(F.interpolate(d2, size=e1_s.shape[2:], mode='bilinear', align_corners=False))
        d3_d1 = self.conv_d3_d1(F.interpolate(d3, size=e1_s.shape[2:], mode='bilinear', align_corners=False))
        d4_d1 = self.conv_d4_d1(F.interpolate(d4, size=e1_s.shape[2:], mode='bilinear', align_corners=False))
        e5_d1 = self.conv_e5_d1(F.interpolate(e5_s, size=e1_s.shape[2:], mode='bilinear', align_corners=False))
        d1 = self.conv_d1(torch.cat([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1], dim=1))

        out = self.out_conv(d1)
        return out
