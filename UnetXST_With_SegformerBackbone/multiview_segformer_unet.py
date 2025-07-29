import torch
import torch.nn as nn
from transformers import SegformerModel
from spatial_transformer import SpatialTransformerFixed
from homography import H_front, H_rear, H_left, H_right
from palette import n_classes_label
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
    def forward(self, x):
        return self.conv(x)

class ViewAttentionFusion(nn.Module):
    """Attention fusion theo chiều view: học trọng số cho từng view tại mỗi spatial location, weighted sum."""
    def __init__(self, n_views, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.n_views = n_views
        self.in_ch = in_ch
        self.out_ch = out_ch
        # Conv3d: [B, C, n_views, H, W] -> [B, 1, n_views, H, W]
        self.attn_conv = nn.Conv3d(in_ch, 1, kernel_size=1)
        self.out_conv = ConvBNReLU(in_ch, out_ch, dropout)

    def forward(self, x):
        # x: [B, n_views, C, H, W]
        x_perm = x.permute(0, 2, 1, 3, 4)  # [B, C, n_views, H, W]
        attn = self.attn_conv(x_perm)       # [B, 1, n_views, H, W]
        attn = attn.squeeze(1)              # [B, n_views, H, W]
        attn = torch.softmax(attn, dim=1)   # softmax theo chiều view
        # Weighted sum các view
        fused = (x * attn.unsqueeze(2)).sum(dim=1)  # [B, C, H, W]
        out = self.out_conv(fused)
        return out

class MultiViewSegformerUNet(nn.Module):
    def __init__(self, 
                 segformer_name='nvidia/segformer-b2-finetuned-ade-512-512',
                 n_classes=n_classes_label, 
                 dropout=0.1):
        super().__init__()
        self.n_views = 4
        self.n_classes = n_classes

        # 4 encoder, mỗi encoder là 1 SegFormer backbone riêng biệt
        self.encoders = nn.ModuleList([
            SegformerModel.from_pretrained(segformer_name) for _ in range(self.n_views)
        ])

        # Lấy số channel từng tầng skip từ encoder (không hardcode)
        dummy = torch.randn(1, 3, 128, 256)
        with torch.no_grad():
            feats = self.encoders[0](dummy, output_hidden_states=True).hidden_states
        skip_channels = [f.shape[1] for f in feats[1:]]  # lấy tất cả tầng skip (bỏ patch embedding)
        self.n_skips = len(skip_channels)
        print("Skip channels:", skip_channels)

        # Spatial Transformer cho từng view, từng tầng (n_view x n_skips)
        H_list = [H_front, H_rear, H_left, H_right]
        self.stn = nn.ModuleList([
            nn.ModuleList([
                SpatialTransformerFixed(out_size=None, homography=H_list[view])
                for _ in range(self.n_skips)
            ]) for view in range(self.n_views)
        ])

        # Attention fusion block cho mỗi tầng
        self.fusion_blocks = nn.ModuleList([
            ViewAttentionFusion(self.n_views, skip_channels[i], skip_channels[i], dropout)
            for i in range(self.n_skips)
        ])

        # Decoder UNet-style tổng quát
        features = skip_channels[::-1]  # vd: [256, 160, 64]
        self.decoders = nn.ModuleList()
        for i in range(len(features)-1):
            self.decoders.append(nn.ConvTranspose2d(features[i], features[i+1], kernel_size=2, stride=2))
            # Số channel sau concat = features[i+1] * 2
            self.decoders.append(ConvBNReLU(features[i+1]*2, features[i+1], dropout))
        self.final_layer = nn.Conv2d(features[-1], n_classes, kernel_size=1)

    def forward(self, imgs):
        # imgs: list of 4 tensor [B, 3, H, W]
        B = imgs[0].shape[0]
        # 1. Encode từng view, lấy skip connection
        all_skips = []
        for v in range(self.n_views):
            feats = self.encoders[v](imgs[v], output_hidden_states=True).hidden_states
            skips = list(feats[1:])  # lấy tất cả tầng skip
            all_skips.append(skips)

        # 2. Fusion từng tầng với attention
        fused_skips = []
        for d in range(self.n_skips):
            warped = []
            for v in range(self.n_views):
                feat = all_skips[v][d]
                # Set out_size cho SpatialTransformerFixed nếu chưa set
                if self.stn[v][d].out_h is None or self.stn[v][d].out_w is None:
                    h, w = feat.shape[2:]
                    self.stn[v][d].out_h = h
                    self.stn[v][d].out_w = w
                warped_feat = self.stn[v][d](feat)
                warped.append(warped_feat)
            warped = torch.stack(warped, dim=1)  # [B, n_views, C, H, W]
            fused = self.fusion_blocks[d](warped)
            fused_skips.append(fused)

        # 3. Decoder tổng quát (không hardcode số tầng)
        x = fused_skips[-1]
        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)
            skip_idx = -(i//2 + 2)
            if abs(skip_idx) <= len(fused_skips):
                x = torch.cat([x, fused_skips[skip_idx]], dim=1)
            x = self.decoders[i+1](x)
        x = self.final_layer(x)
        x = F.interpolate(x, size=(128, 256), mode='bilinear', align_corners=False)
        return x

if __name__ == "__main__":
    # Test forward
    B = 2
    imgs = [torch.randn(B, 3, 128, 256) for _ in range(4)]
    model = MultiViewSegformerUNet(
        segformer_name='nvidia/segformer-b2-finetuned-ade-512-512',
        dropout=0.1
    )
    y = model(imgs)
    print("Output shape:", y.shape)  # (B, n_classes, 128, 256)
    print("Fusion block type:", type(model.fusion_blocks[0]))