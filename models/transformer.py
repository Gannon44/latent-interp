import torch
import torch.nn as nn

class TransformerInterp(nn.Module):
    def __init__(self, latent_dim=4, spatial_size=32, d_model=128, nhead=4, num_layers=2):
        """
        This model first applies a convolutional patch embedder to convert the
        input tensor (8,32,32) into patches, processes them with a transformer,
        then upsamples to produce a latent prediction (4,32,32).
        """
        super(TransformerInterp, self).__init__()
        # Patch embedding: convert input from 8 channels to d_model channels.
        # Using kernel_size=4 and stride=4 downsamples spatial dims from 32x32 to 8x8.
        self.patch_embed = nn.Conv2d(8, d_model, kernel_size=4, stride=4)  # -> (B, d_model, 8, 8)
        seq_len = 8 * 8
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # For each patch, predict a latent vector of length latent_dim.
        self.fc = nn.Linear(d_model, latent_dim)
        # Upsampling: from patch grid (8,8) to full resolution (32,32)
        self.up = nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=4, stride=4)
    
    def forward(self, latent1, latent3):
        x = torch.cat([latent1, latent3], dim=1)  # (B,8,32,32)
        x = self.patch_embed(x)  # (B, d_model, 8,8)
        B, d_model, H, W = x.shape
        x = x.view(B, d_model, -1).transpose(1, 2)  # (B, seq_len, d_model)
        x = x + self.pos_embed  # add positional info
        # Transformer expects (seq_len, B, d_model)
        x = x.transpose(0, 1)
        x = self.transformer(x)  # (seq_len, B, d_model)
        x = x.transpose(0, 1)  # (B, seq_len, d_model)
        # Predict latent vector for each patch.
        x = self.fc(x)  # (B, seq_len, latent_dim)
        # Reshape into spatial grid.
        x = x.transpose(1, 2).view(B, -1, H, W)  # (B, latent_dim, 8,8)
        out = self.up(x)  # (B, latent_dim, 32,32)
        return out