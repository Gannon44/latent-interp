import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from tqdm import tqdm

# Custom dataset to load images from triplet directories
class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with subdirectories, each containing frame1.jpg, frame2.jpg, frame3.jpg.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        # Loop over each subdirectory and add available frame*.jpg files to the list
        for split in os.listdir(root_dir):
            split_dir = os.path.join(root_dir,split)
            print(split)
            print(split_dir)
            for subdir in os.listdir(split_dir):
                full_subdir = os.path.join(root_dir, split, subdir)
                if os.path.isdir(full_subdir):
                    # Use all 3 frames per triplet as independent training samples
                    for i in range(1, 4):
                        img_path = os.path.join(full_subdir, f"frame{i}.jpg")
                        if os.path.exists(img_path):
                            self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Function to compute KL divergence loss for the VAE
def kl_divergence(mu, logvar):
    # Standard VAE KL divergence between N(mu, sigma) and N(0,1)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def main():
    # Configuration parameters
    root_dir = "/data/ggonsior/atd12k"
    batch_size = 8
    epochs = 10
    learning_rate = 1e-4
    image_size = (256, 256)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    if device == 'cpu':
        print(torch.cuda.is_available())
        raise "no gpu"
    # Define image transformations (resize, convert to tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Normalize to [-1, 1]; adjust if your model requires different normalization.
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Create dataset and dataloader
    train_dataset = AnimeDataset(root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load the pre-trained VAE ("kl-f8-anime2") from diffusers
    print("Loading pre-trained VAE...")
    vae = AutoencoderKL.from_pretrained("bullhug/kl-f8-anime2")
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.to(device)
    vae.train()  # Set the model to training mode

    # Set up optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    # Training loop with tqdm for progress tracking
    for epoch in range(epochs):
        total_loss, total_recon_loss, total_kl_loss = 0.0, 0.0, 0.0

        # Outer tqdm for epoch progress
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", position=0, leave=True) as pbar_epoch:
            for batch in tqdm(train_loader, desc="Batch Progress", position=1, leave=False):
                batch = batch.to(device)
                optimizer.zero_grad()

                posterior = vae.encode(batch).latent_dist
                z = posterior.sample()
                mu = posterior.mean
                sigma = posterior.logvar

                # del posterior
                # gc.collect()
                # torch.cuda.empty_cache()

                reconstruction = vae.decode(z).sample

                recon_loss = F.mse_loss(reconstruction, batch)
                kl_loss = kl_divergence(mu, sigma) / batch.size(0)
                loss = recon_loss + kl_loss

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Accumulate losses
                total_loss += loss.item() * batch.size(0)
                total_recon_loss += recon_loss.item() * batch.size(0)
                total_kl_loss += kl_loss.item() * batch.size(0)

                # Update batch progress bar
                pbar_epoch.update(1)
                pbar_epoch.set_postfix({"Loss": loss.item(), "KL": kl_loss.item(), "Recon": recon_loss.item()})

                # del batch, reconstruction, mu, sigma, z
                # gc.collect()
                # torch.cuda.empty_cache()

        # Compute epoch losses
        avg_loss = total_loss / len(train_dataset)
        avg_recon = total_recon_loss / len(train_dataset)
        avg_kl = total_kl_loss / len(train_dataset)

        torch.save(vae.state_dict(), f"checkpoints/vae_r2_epoch{epoch+1}.pth")


if __name__ == "__main__":
    main()
