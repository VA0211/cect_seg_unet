import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Block

class UNETR(nn.Module):
    def __init__(self, in_chns=1, class_num=4, img_size=(256, 256)):
        super().__init__()
        self.config = {
            "image_height": img_size[0],
            "image_width": img_size[1],
            "num_channels": in_chns,
            "patch_height": 16,
            "patch_width": 16,
            "num_patches": (img_size[0] * img_size[1]) // (16 * 16),
            "num_layers": 12,
            "hidden_dim": 768,
            "mlp_dim": 3072,
            "dropout_rate": 0.0,
        }
        self.final_conv = nn.Conv2d(64, class_num, 1)  

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=16,
            in_chans=self.config['num_channels'],
            embed_dim=self.config['hidden_dim']
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.config['hidden_dim']))
        self.dropout = nn.Dropout(p=self.config['dropout_rate'])

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=self.config['hidden_dim'],
                num_heads=num_heads,
                mlp_ratio=self.config['mlp_dim']/self.config['hidden_dim'],
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                drop=self.config['dropout_rate'],
                attn_drop=self.config['dropout_rate'],
                drop_path=0.0
            ) for _ in range(self.config['num_layers'])
        ])
        self.norm = nn.LayerNorm(self.config['hidden_dim'], eps=1e-6)

        # Encoder projections at multiple layers
        self.encoder1 = nn.Conv3d(self.config['hidden_dim'], feature_size, 1)
        self.encoder2 = nn.Conv3d(self.config['hidden_dim'], feature_size * 2, 1)
        self.encoder3 = nn.Conv3d(self.config['hidden_dim'], feature_size * 4, 1)
        self.encoder4 = nn.Conv3d(self.config['hidden_dim'], feature_size * 8, 1)

        # Decoder
        self.decoder4 = self._upsample_block(feature_size * 8, feature_size * 4)
        self.decoder3 = self._upsample_block(feature_size * 4, feature_size * 2)
        self.decoder2 = self._upsample_block(feature_size * 2, feature_size)
        self.decoder1 = self._upsample_block(feature_size, feature_size)

        self.final_conv = nn.Conv2d(feature_size, out_channels, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input size must be {self.img_size}x{self.img_size}"

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, hidden_size]
        x += self.pos_embed
        x = self.dropout(x)

        hidden_states = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in {3, 6, 9, 11}:
                hidden_states.append(x)

        x = self.norm(x)
        hidden_states.append(x)

        # Reshape and extract features
        d = int(self.img_size // 16)
        feats = [rearrange(h, 'b (h w) c -> b c h w', h=d, w=d) for h in hidden_states]

        e1 = self.encoder1(feats[0].unsqueeze(2)).squeeze(2)
        e2 = self.encoder2(feats[1].unsqueeze(2)).squeeze(2)
        e3 = self.encoder3(feats[2].unsqueeze(2)).squeeze(2)
        e4 = self.encoder4(feats[3].unsqueeze(2)).squeeze(2)

        # Decoder path
        d4 = self.decoder4(e4)
        d3 = self.decoder3(d4 + e3)
        d2 = self.decoder2(d3 + e2)
        d1 = self.decoder1(d2 + e1)

        out = self.final_conv(d1)
        return out