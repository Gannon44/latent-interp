import torch
import torch.nn as nn
import torch.nn.functional as F

class NNModel(nn.Module):
    def __init__(self, latent_dim=4, spatial_size=32):
        """
        Input: concatenation of latent1 and latent3 -> shape (batch, 8, 32, 32)
        Output: prediction for latent2 -> shape (batch, 4, 32, 32)
        """
        super(NNModel, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, latent_dim, kernel_size=3, padding=1)
        
    def forward(self, latent1, latent3):
        x = torch.cat([latent1, latent3], dim=1)  # (B, 8, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = self.conv4(x)  # (B, 4, 32, 32)
        return out
