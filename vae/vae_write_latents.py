import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from tqdm import tqdm

# Define the transform used for encoding (must match your training transform)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Device for encoding (adjust as needed)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load your trained VAE; adjust the pretrained model identifier if necessary.
vae = AutoencoderKL.from_pretrained("bullhug/kl-f8-anime2")
# vae.load_state_dict(torch.load("/code/ggonsior/487/latent-interp/checkpoints/vae_r2_epoch10.pth", map_location=device))
vae.to(device)
vae.eval()

# Base directory for your splits
base_dir = "/data/ggonsior/atd12k"
splits = ["train", "val", "test"]

# Gather all subdirectories across splits (each subdirectory should contain the three frames)
subdirs = []
for split in splits:
    split_dir = os.path.join(base_dir, split)
    if not os.path.isdir(split_dir):
        continue
    for subdir in os.listdir(split_dir):
        full_subdir = os.path.join(split_dir, subdir)
        if os.path.isdir(full_subdir):
            subdirs.append(full_subdir)

print(f"Found {len(subdirs)} subdirectories to process.")

# Process each subdirectory with a progress bar.
for subdir in tqdm(subdirs, desc="Processing subdirectories"):
    # For each subdir, prepare a list of frame file paths.
    frame_paths = []
    for i in range(1, 4):
        path = os.path.join(subdir, f"frame{i}.jpg")
        if os.path.exists(path):
            frame_paths.append(path)
    
    # If no frames found, skip this subdir.
    if len(frame_paths) == 0:
        continue

    # Load all frames and apply the transform.
    imgs = []
    for path in frame_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            imgs.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if len(imgs) == 0:
        continue

    # Stack images to form a batch (shape: [batch_size, 3, 256, 256])
    batch = torch.stack(imgs, dim=0).to(device)
    
    # Encode the batch and use the latent mean as representation.
    with torch.no_grad():
        latent = vae.encode(batch).latent_dist.mean  # shape: [batch_size, C, H, W]
    
    # Save each latent representation as a .npy file
    latent = latent.cpu().numpy()  # convert to numpy array
    for i in range(latent.shape[0]):
        latent_path = os.path.join(subdir, f"latent{i+1}.npy")
        np.save(latent_path, latent[i])
