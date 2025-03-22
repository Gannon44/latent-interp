import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, latent_dim=4, spatial_size=32):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(8 * spatial_size * spatial_size, latent_dim * spatial_size * spatial_size)
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
    
    def forward(self, latent1, latent3):
        x = torch.cat([latent1, latent3], dim=1)  # (B,8,32,32)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        out = out.view(x.size(0), 4, self.spatial_size, self.spatial_size)
        return out