import torch
import torch.nn as nn

class UNetInterp(nn.Module):
    def __init__(self, latent_dim=4, in_channels=8):
        """
        A UNet architecture that takes input (B, 8,32,32) and produces output (B, 4,32,32).
        """
        super(UNetInterp, self).__init__()
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder path
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.out_conv = nn.Conv2d(32, latent_dim, kernel_size=1)
    
    def forward(self, latent1, latent3):
        x = torch.cat([latent1, latent3], dim=1)  # (B,8,32,32)
        e1 = self.enc1(x)        # (B,32,32,32)
        e2 = self.enc2(e1)       # (B,64,16,16)
        e3 = self.enc3(e2)       # (B,128,8,8)
        b = self.bottleneck(e3)  # (B,128,8,8)
        d3 = self.dec3(b)        # (B,64,16,16)
        d2 = self.dec2(d3)       # (B,32,32,32)
        out = self.out_conv(d2)  # (B, latent_dim,32,32)
        return out