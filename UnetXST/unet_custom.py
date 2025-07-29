import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_transformer import SpatialTransformerFixed
from homography import H_front, H_rear, H_left, H_right
from palette import n_classes_label

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class encoding_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoding_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetCustom(nn.Module):
    def __init__(self, in_channels=3, out_channels=n_classes_label, features=[64, 128, 256, 512]):
        super(UNetCustom, self).__init__()
        self.n_views = 4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Encoder blocks (dùng chung cho các view)
        self.conv1 = encoding_block(in_channels, features[0])
        self.conv2 = encoding_block(features[0], features[1])
        self.conv3 = encoding_block(features[1], features[2])
        self.conv4 = encoding_block(features[2], features[3])
        self.bottleneck = encoding_block(features[3], features[3]*2)
        # Spatial Transformer cho từng view, từng tầng skip
        H_list = [H_front, H_rear, H_left, H_right]
        self.n_skips = 4  # số tầng skip
        self.stn = nn.ModuleList([
            nn.ModuleList([
                SpatialTransformerFixed(out_size=None, homography=H_list[view])
                for _ in range(self.n_skips)
            ]) for view in range(self.n_views)
        ])
        # Decoder blocks (số channel sẽ lớn hơn do concat views)
        self.tconv1 = nn.ConvTranspose2d(features[3]*2*self.n_views, features[3], kernel_size=2, stride=2)
        self.conv5 = encoding_block(features[3]*self.n_views + features[3], features[3])
        self.tconv2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.conv6 = encoding_block(features[2]*self.n_views + features[2], features[2])
        self.tconv3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.conv7 = encoding_block(features[1]*self.n_views + features[1], features[1])
        self.tconv4 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.conv8 = encoding_block(features[0]*self.n_views + features[0], features[0])
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, imgs):
        # imgs: list of 4 tensor [B, C, H, W]
        assert isinstance(imgs, list) and len(imgs) == self.n_views, "Input must be list of 4 tensors [B, C, H, W]"
        # 1. Encode từng view, lấy skip connection
        all_skips = [[] for _ in range(self.n_views)]
        enc_outs = []
        for v in range(self.n_views):
            x = imgs[v]
            x1 = self.conv1(x)
            all_skips[v].append(x1)
            x2 = self.pool(x1)
            x2 = self.conv2(x2)
            all_skips[v].append(x2)
            x3 = self.pool(x2)
            x3 = self.conv3(x3)
            all_skips[v].append(x3)
            x4 = self.pool(x3)
            x4 = self.conv4(x4)
            all_skips[v].append(x4)
            x5 = self.pool(x4)
            enc_outs.append(self.bottleneck(x5))
        # 2. Áp dụng spatial transformer cho từng skip connection trước khi fusion
        fused_skips = []
        n_skips = len(all_skips[0])  # 4
        for i in range(n_skips):
            # Apply spatial transformer cho từng view, từng tầng
            transformed = []
            for v in range(self.n_views):
                stn = self.stn[v][i]
                feat = all_skips[v][i]
                stn.out_h, stn.out_w = feat.shape[-2], feat.shape[-1]
                transformed.append(stn(feat))
            fused = torch.cat(transformed, dim=1)
            fused_skips.append(fused)
        # Fusion bottleneck (không cần spatial transformer ở bottleneck)
        fused_bottleneck = torch.cat(enc_outs, dim=1)  # [B, features[3]*2*4, H, W]
        # 3. Decoder như UNet, nhưng số channel sẽ lớn hơn
        skip_connections = fused_skips[::-1]
        x = self.tconv1(fused_bottleneck)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x

if __name__ == "__main__":
    # Example usage and parameter count
    model = UNetCustom(in_channels=3, out_channels=8)
    x = torch.randn(2, 3, 96, 144)
    y = model(x)
    print("Output shape:", y.shape)  # (2, 8, 96, 144)
    print("Trainable parameters:", count_parameters(model))