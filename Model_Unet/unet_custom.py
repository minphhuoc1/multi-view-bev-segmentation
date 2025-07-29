import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels=10, out_channels=10, features=[64, 128, 256, 512]):
        super(UNetCustom, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = encoding_block(in_channels, features[0])
        self.conv2 = encoding_block(features[0], features[1])
        self.conv3 = encoding_block(features[1], features[2])
        self.conv4 = encoding_block(features[2], features[3])
        self.bottleneck = encoding_block(features[3], features[3]*2)
        self.tconv1 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.conv5 = encoding_block(features[3]*2, features[3])
        self.tconv2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.conv6 = encoding_block(features[2]*2, features[2])
        self.tconv3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.conv7 = encoding_block(features[1]*2, features[1])
        self.tconv4 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.conv8 = encoding_block(features[0]*2, features[0])
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x1 = self.conv1(x)
        skip_connections.append(x1)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        skip_connections.append(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        skip_connections.append(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        skip_connections.append(x4)
        x5 = self.pool(x4)
        x5 = self.bottleneck(x5)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x5)
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
    # Kiểm tra shape của 1 batch
    sample_img, sample_mask = train_dataset[0]
    print("Sample image shape:", sample_img.shape)  # (3, 128, 256)
    print("Sample mask shape:", sample_mask.shape)  # (128, 256)

    # Kiểm tra forward pass
    model = UNetCustom(in_channels=3, out_channels=7)
    x = sample_img.unsqueeze(0)  # [1, 3, 128, 256]
    y = model(x)
    print("Model output shape:", y.shape)  # [1, 7, 128, 256]
