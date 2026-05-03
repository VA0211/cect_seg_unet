import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class ESCA(nn.Module):
    """
    Efficient Small Channel Attention (ESCA) block as described in the STS-Net paper.
    """
    def __init__(self, in_channels, b=1, gamma=2):
        super(ESCA, self).__init__()
        kernel_size = int(abs((math.log(in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) for the DeepLabV3 decoder.
    """
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

class STSNet(nn.Module):
    """
    STS-Net (ResNeXt50 + ESCA + DeepLabV3)
    """
    def __init__(self, in_channels=1, num_classes=2):
        super(STSNet, self).__init__()
        
        # Load the ResNeXt50-32x4d backbone
        resnext = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        
        # Override the first conv layer to accept 1 channel instead of 3
        if in_channels != 3:
            resnext.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.encoder1 = nn.Sequential(resnext.conv1, resnext.bn1, resnext.relu, resnext.maxpool)
        self.encoder2 = resnext.layer1
        self.esca2 = ESCA(256)
        
        self.encoder3 = resnext.layer2
        self.esca3 = ESCA(512)
        
        self.encoder4 = resnext.layer3
        self.esca4 = ESCA(1024)
        
        self.encoder5 = resnext.layer4
        self.esca5 = ESCA(2048)
        
        # DeepLabV3 Decoder (ASPP)
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        
        # Final segmentation head
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        
        e1 = self.encoder1(x)
        e2 = self.esca2(self.encoder2(e1))
        e3 = self.esca3(self.encoder3(e2))
        e4 = self.esca4(self.encoder4(e3))
        e5 = self.esca5(self.encoder5(e4))
        
        aspp_out = self.aspp(e5)
        out = self.classifier(aspp_out)
        
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        return out