import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import numpy as np
from tqdm import tqdm

class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for split in os.listdir(root_dir):
            split_dir = os.path.join(root_dir, split)
            for subdir in os.listdir(split_dir):
                full_subdir = os.path.join(split_dir, subdir)
                if os.path.isdir(full_subdir):
                    for i in range(1, 4):
                        path = os.path.join(full_subdir, f"frame{i}.jpg")
                        if os.path.exists(path):
                            self.image_paths.append(path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

def denormalize(tensor):
    # Reverse normalization from [-1, 1] to [0, 1]
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def main():
    root_dir = "/data/ggonsior/atd12k"
    batch_size = 8
    image_size = (256, 256)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    
    dataset = AnimeDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    vae = AutoencoderKL.from_pretrained("bullhug/kl-f8-anime2")
    # vae.load_state_dict(torch.load("/code/ggonsior/487/latent-interp/checkpoints/vae_r2_epoch10.pth", map_location=device))
    vae.to(device)
    vae.eval()
    
    mse_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    count = 0
    
    with torch.no_grad():
        for batch, paths in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            posterior = vae.encode(batch).latent_dist
            z = posterior.sample()
            recon = vae.decode(z).sample
            recon = denormalize(recon).cpu().numpy()
            gt = denormalize(batch).cpu().numpy()
            for i in range(recon.shape[0]):
                mse = np.mean((recon[i] - gt[i]) ** 2)
                psnr = compare_psnr(gt[i].transpose(1,2,0), recon[i].transpose(1,2,0), data_range=1.0)
                ssim = compare_ssim(gt[i].transpose(1,2,0), recon[i].transpose(1,2,0), win_size=7, channel_axis=-1, data_range=1.0)
                mse_total += mse
                psnr_total += psnr
                ssim_total += ssim
                count += 1
                
    print(f"Average MSE: {mse_total/count:.4f}")
    print(f"Average PSNR: {psnr_total/count:.4f}")
    print(f"Average SSIM: {ssim_total/count:.4f}")

if __name__ == "__main__":
    main()
