import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import AutoModel
import timm

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if act else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Resnet50_Unet3plus(nn.Module):
    def __init__(self, num_classes=2, base_filters=64, in_ch=1, pretrained=True):
        super().__init__()
        # ResNet50 Encoder
        resnet = resnet50(pretrained=pretrained)
        if in_ch != 3:
            weight = resnet.conv1.weight.clone()
            resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if in_ch == 1:
                resnet.conv1.weight.data = weight.sum(dim=1, keepdim=True)
            else:
                resnet.conv1.weight.data[:, :3] = weight

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        f = base_filters
        
        # Standardize encoder outputs to base_filters channels
        self.conv_encode1 = ConvBlock(64, f)
        self.conv_encode2 = ConvBlock(256, f)
        self.conv_encode3 = ConvBlock(512, f)
        self.conv_encode4 = ConvBlock(1024, f)
        self.conv_encode5 = ConvBlock(2048, f)

        # Decoder convolution blocks
        self.conv_e1_d4 = ConvBlock(f, f)
        self.conv_e2_d4 = ConvBlock(f, f)
        self.conv_e3_d4 = ConvBlock(f, f)
        self.conv_e4_d4 = ConvBlock(f, f)
        self.conv_e5_d4 = ConvBlock(f, f)
        self.conv_d4 = ConvBlock(f * 5, f * 5)

        self.conv_e1_d3 = ConvBlock(f, f)
        self.conv_e2_d3 = ConvBlock(f, f)
        self.conv_e3_d3 = ConvBlock(f, f)
        self.conv_d4_d3 = ConvBlock(f * 5, f)
        self.conv_e5_d3 = ConvBlock(f, f)
        self.conv_d3 = ConvBlock(f * 5, f * 5)

        self.conv_e1_d2 = ConvBlock(f, f)
        self.conv_e2_d2 = ConvBlock(f, f)
        self.conv_d3_d2 = ConvBlock(f * 5, f)
        self.conv_d4_d2 = ConvBlock(f * 5, f)
        self.conv_e5_d2 = ConvBlock(f, f)
        self.conv_d2 = ConvBlock(f * 5, f * 5)

        self.conv_e1_d1 = ConvBlock(f, f)
        self.conv_d2_d1 = ConvBlock(f * 5, f)
        self.conv_d3_d1 = ConvBlock(f * 5, f)
        self.conv_d4_d1 = ConvBlock(f * 5, f)
        self.conv_e5_d1 = ConvBlock(f, f)
        self.conv_d1 = ConvBlock(f * 5, f * 5)

        self.out_conv = nn.Conv2d(f * 5, num_classes, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)      # [N, 64, H/2, W/2]
        e2 = self.enc2(e1)     # [N, 256, H/4, W/4]
        e3 = self.enc3(e2)     # [N, 512, H/8, W/8]
        e4 = self.enc4(e3)     # [N,1024,H/16,W/16]
        e5 = self.enc5(e4)     # [N,2048,H/32,W/32]

        e1_c = self.conv_encode1(e1)
        e2_c = self.conv_encode2(e2)
        e3_c = self.conv_encode3(e3)
        e4_c = self.conv_encode4(e4)
        e5_c = self.conv_encode5(e5)

        # Decoder 4
        size4 = e4_c.shape[2:]
        e1_d4 = self.conv_e1_d4(F.adaptive_max_pool2d(e1_c, size4))
        e2_d4 = self.conv_e2_d4(F.adaptive_max_pool2d(e2_c, size4))
        e3_d4 = self.conv_e3_d4(F.adaptive_max_pool2d(e3_c, size4))
        e4_d4 = self.conv_e4_d4(e4_c)
        e5_d4 = self.conv_e5_d4(F.interpolate(e5_c, size=size4, mode='bilinear', align_corners=False))
        d4 = self.conv_d4(torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], dim=1))

        # Decoder 3
        size3 = e3_c.shape[2:]
        e1_d3 = self.conv_e1_d3(F.adaptive_max_pool2d(e1_c, size3))
        e2_d3 = self.conv_e2_d3(F.adaptive_max_pool2d(e2_c, size3))
        e3_d3 = self.conv_e3_d3(e3_c)
        d4_d3 = self.conv_d4_d3(F.interpolate(d4, size=size3, mode='bilinear', align_corners=False))
        e5_d3 = self.conv_e5_d3(F.interpolate(e5_c, size=size3, mode='bilinear', align_corners=False))
        d3 = self.conv_d3(torch.cat([e1_d3, e2_d3, e3_d3, d4_d3, e5_d3], dim=1))

        # Decoder 2
        size2 = e2_c.shape[2:]
        e1_d2 = self.conv_e1_d2(F.adaptive_max_pool2d(e1_c, size2))
        e2_d2 = self.conv_e2_d2(e2_c)
        d3_d2 = self.conv_d3_d2(F.interpolate(d3, size=size2, mode='bilinear', align_corners=False))
        d4_d2 = self.conv_d4_d2(F.interpolate(d4, size=size2, mode='bilinear', align_corners=False))
        e5_d2 = self.conv_e5_d2(F.interpolate(e5_c, size=size2, mode='bilinear', align_corners=False))
        d2 = self.conv_d2(torch.cat([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2], dim=1))

        # Decoder 1 (finest scale)
        size1 = e1_c.shape[2:]
        e1_d1 = self.conv_e1_d1(e1_c)
        d2_d1 = self.conv_d2_d1(F.interpolate(d2, size=size1, mode='bilinear', align_corners=False))
        d3_d1 = self.conv_d3_d1(F.interpolate(d3, size=size1, mode='bilinear', align_corners=False))
        d4_d1 = self.conv_d4_d1(F.interpolate(d4, size=size1, mode='bilinear', align_corners=False))
        e5_d1 = self.conv_e5_d1(F.interpolate(e5_c, size=size1, mode='bilinear', align_corners=False))
        d1 = self.conv_d1(torch.cat([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1], dim=1))

        out = self.out_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        # raw logits output - do NOT apply softmax/sigmoid here to keep consistent with the training code
        return out

class MambaUNet3plus(nn.Module):
    def __init__(
        self, 
        num_classes=2, 
        base_filters=64, 
        pretrained=True, 
        in_ch=1, 
        backbone_ckpt="nvidia/MambaVision-T-1K"
    ):
        super().__init__()
        # MambaVision as backbone
        self.mamba = AutoModel.from_pretrained(backbone_ckpt, trust_remote_code=True)
        if pretrained:
            self.mamba.eval()
        # MambaVision stages output: features[0]:80, features[1]:160, features[2]:320, features[3]:640 channels
        channels = [80, 160, 320, 640]
        f = base_filters  # bottleneck feature channels

        # Feature normalization as usual
        self.conv_encode1 = ConvBlock(channels[0], f)
        self.conv_encode2 = ConvBlock(channels[1], f)
        self.conv_encode3 = ConvBlock(channels[2], f)
        self.conv_encode4 = ConvBlock(channels[3], f)

        # Decoder conv blocks (same as before; no change needed)
        self.conv_e1_d4 = ConvBlock(f, f)
        self.conv_e2_d4 = ConvBlock(f, f)
        self.conv_e3_d4 = ConvBlock(f, f)
        self.conv_e4_d4 = ConvBlock(f, f)
        self.conv_d4 = ConvBlock(f * 4, f * 4)

        self.conv_e1_d3 = ConvBlock(f, f)
        self.conv_e2_d3 = ConvBlock(f, f)
        self.conv_e3_d3 = ConvBlock(f, f)
        self.conv_d4_d3 = ConvBlock(f * 4, f)
        self.conv_d3 = ConvBlock(f * 4, f * 4)

        self.conv_e1_d2 = ConvBlock(f, f)
        self.conv_e2_d2 = ConvBlock(f, f)
        self.conv_d3_d2 = ConvBlock(f * 4, f)
        self.conv_d4_d2 = ConvBlock(f * 4, f)
        self.conv_d2 = ConvBlock(f * 4, f * 4)

        self.conv_e1_d1 = ConvBlock(f, f)
        self.conv_d2_d1 = ConvBlock(f * 4, f)
        self.conv_d3_d1 = ConvBlock(f * 4, f)
        self.conv_d4_d1 = ConvBlock(f * 4, f)
        self.conv_d1 = ConvBlock(f * 4, f * 4)

        self.out_conv = nn.Conv2d(f * 4, num_classes, 3, padding=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Get hierarchical features from MambaVision
        with torch.no_grad():  # Backbone in eval mode by default
            _, features = self.mamba(x)
        e1, e2, e3, e4 = features

        # Standardize channel dims
        e1_c = self.conv_encode1(e1)
        e2_c = self.conv_encode2(e2)
        e3_c = self.conv_encode3(e3)
        e4_c = self.conv_encode4(e4)

        # Decoder 4 (lowest resolution, deepest)
        size4 = e4_c.shape[2:]
        e1_d4 = self.conv_e1_d4(F.adaptive_max_pool2d(e1_c, size4))
        e2_d4 = self.conv_e2_d4(F.adaptive_max_pool2d(e2_c, size4))
        e3_d4 = self.conv_e3_d4(F.adaptive_max_pool2d(e3_c, size4))
        e4_d4 = self.conv_e4_d4(e4_c)
        d4 = self.conv_d4(torch.cat([e1_d4, e2_d4, e3_d4, e4_d4], dim=1))

        # Decoder 3
        size3 = e3_c.shape[2:]
        e1_d3 = self.conv_e1_d3(F.adaptive_max_pool2d(e1_c, size3))
        e2_d3 = self.conv_e2_d3(F.adaptive_max_pool2d(e2_c, size3))
        e3_d3 = self.conv_e3_d3(e3_c)
        d4_d3 = self.conv_d4_d3(F.interpolate(d4, size=size3, mode='bilinear', align_corners=False))
        d3 = self.conv_d3(torch.cat([e1_d3, e2_d3, e3_d3, d4_d3], dim=1))

        # Decoder 2
        size2 = e2_c.shape[2:]
        e1_d2 = self.conv_e1_d2(F.adaptive_max_pool2d(e1_c, size2))
        e2_d2 = self.conv_e2_d2(e2_c)
        d3_d2 = self.conv_d3_d2(F.interpolate(d3, size=size2, mode='bilinear', align_corners=False))
        d4_d2 = self.conv_d4_d2(F.interpolate(d4, size=size2, mode='bilinear', align_corners=False))
        d2 = self.conv_d2(torch.cat([e1_d2, e2_d2, d3_d2, d4_d2], dim=1))

        # Decoder 1 (finest scale)
        size1 = e1_c.shape[2:]
        e1_d1 = self.conv_e1_d1(e1_c)
        d2_d1 = self.conv_d2_d1(F.interpolate(d2, size=size1, mode='bilinear', align_corners=False))
        d3_d1 = self.conv_d3_d1(F.interpolate(d3, size=size1, mode='bilinear', align_corners=False))
        d4_d1 = self.conv_d4_d1(F.interpolate(d4, size=size1, mode='bilinear', align_corners=False))
        d1 = self.conv_d1(torch.cat([e1_d1, d2_d1, d3_d1, d4_d1], dim=1))

        out = self.out_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

class Transformer_UNet3plus(nn.Module):
    def __init__(
        self, 
        num_classes=2, 
        base_filters=64, 
        in_ch=1, 
        backbone="swin_small_patch4_window7_224", 
        pretrained=True
    ):
        super().__init__()
        # Swin Small backbone (always expects in_chans=3)
        self.encoder = timm.create_model(
            backbone, features_only=True, pretrained=pretrained, in_chans=3
        )
        encoder_channels = self.encoder.feature_info.channels()
        f = base_filters

        self.conv_encode1 = ConvBlock(encoder_channels[0], f)
        self.conv_encode2 = ConvBlock(encoder_channels[1], f)
        self.conv_encode3 = ConvBlock(encoder_channels[2], f)
        self.conv_encode4 = ConvBlock(encoder_channels[3], f)

        self.conv_e1_d4 = ConvBlock(f, f)
        self.conv_e2_d4 = ConvBlock(f, f)
        self.conv_e3_d4 = ConvBlock(f, f)
        self.conv_e4_d4 = ConvBlock(f, f)
        self.conv_d4 = ConvBlock(f * 4, f * 4)

        self.conv_e1_d3 = ConvBlock(f, f)
        self.conv_e2_d3 = ConvBlock(f, f)
        self.conv_e3_d3 = ConvBlock(f, f)
        self.conv_d4_d3 = ConvBlock(f * 4, f)
        self.conv_d3 = ConvBlock(f * 4, f * 4)

        self.conv_e1_d2 = ConvBlock(f, f)
        self.conv_e2_d2 = ConvBlock(f, f)
        self.conv_d3_d2 = ConvBlock(f * 4, f)
        self.conv_d4_d2 = ConvBlock(f * 4, f)
        self.conv_d2 = ConvBlock(f * 4, f * 4)

        self.conv_e1_d1 = ConvBlock(f, f)
        self.conv_d2_d1 = ConvBlock(f * 4, f)
        self.conv_d3_d1 = ConvBlock(f * 4, f)
        self.conv_d4_d1 = ConvBlock(f * 4, f)
        self.conv_d1 = ConvBlock(f * 4, f * 4)

        self.out_conv = nn.Conv2d(f * 4, num_classes, 3, padding=1)

        self.in_ch = in_ch  # record for use in forward

    def forward(self, x):
        # Ensure 3 channel input for Swin backbone
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3, :, :]  # Truncate to 3 chans if >3

        feats = self.encoder(x)
        # Patch: ensure all features are NCHW
        for i in range(len(feats)):
            if feats[i].dim() == 4 and feats[i].shape[1] < feats[i].shape[-1]:
                # Detected NHWC, convert to NCHW
                feats[i] = feats[i].permute(0, 3, 1, 2).contiguous()
        
        e1_c = self.conv_encode1(feats[0])
        e2_c = self.conv_encode2(feats[1])
        e3_c = self.conv_encode3(feats[2])
        e4_c = self.conv_encode4(feats[3])

        # Decoder 4
        size4 = e4_c.shape[2:]
        e1_d4 = self.conv_e1_d4(F.adaptive_max_pool2d(e1_c, size4))
        e2_d4 = self.conv_e2_d4(F.adaptive_max_pool2d(e2_c, size4))
        e3_d4 = self.conv_e3_d4(F.adaptive_max_pool2d(e3_c, size4))
        e4_d4 = self.conv_e4_d4(e4_c)
        d4 = self.conv_d4(torch.cat([e1_d4, e2_d4, e3_d4, e4_d4], dim=1))

        # Decoder 3
        size3 = e3_c.shape[2:]
        e1_d3 = self.conv_e1_d3(F.adaptive_max_pool2d(e1_c, size3))
        e2_d3 = self.conv_e2_d3(F.adaptive_max_pool2d(e2_c, size3))
        e3_d3 = self.conv_e3_d3(e3_c)
        d4_d3 = self.conv_d4_d3(F.interpolate(d4, size=size3, mode='bilinear', align_corners=False))
        d3 = self.conv_d3(torch.cat([e1_d3, e2_d3, e3_d3, d4_d3], dim=1))

        # Decoder 2
        size2 = e2_c.shape[2:]
        e1_d2 = self.conv_e1_d2(F.adaptive_max_pool2d(e1_c, size2))
        e2_d2 = self.conv_e2_d2(e2_c)
        d3_d2 = self.conv_d3_d2(F.interpolate(d3, size=size2, mode='bilinear', align_corners=False))
        d4_d2 = self.conv_d4_d2(F.interpolate(d4, size=size2, mode='bilinear', align_corners=False))
        d2 = self.conv_d2(torch.cat([e1_d2, e2_d2, d3_d2, d4_d2], dim=1))

        # Decoder 1 (finest scale)
        size1 = e1_c.shape[2:]
        e1_d1 = self.conv_e1_d1(e1_c)
        d2_d1 = self.conv_d2_d1(F.interpolate(d2, size=size1, mode='bilinear', align_corners=False))
        d3_d1 = self.conv_d3_d1(F.interpolate(d3, size=size1, mode='bilinear', align_corners=False))
        d4_d1 = self.conv_d4_d1(F.interpolate(d4, size=size1, mode='bilinear', align_corners=False))
        d1 = self.conv_d1(torch.cat([e1_d1, d2_d1, d3_d1, d4_d1], dim=1))

        out = self.out_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out  # [B, num_classes, H, W], raw logits