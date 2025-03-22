#!/usr/bin/env python
import argparse
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from models import MODEL_DICT
from utils import save_checkpoint, EarlyStopper, plot_metrics

class LatentDataset(Dataset):
    """
    Dataset that loads latent representations saved as .npy files.
    Expects each subdirectory to contain latent1.npy, latent2.npy, latent3.npy.
    """
    def __init__(self, base_dir):
        self.subdirs = []
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(base_dir, split)
            for d in os.listdir(split_dir):
                full_d = os.path.join(split_dir, d)
                if os.path.isdir(full_d):
                    # Only include if all latent files exist.
                    files = [os.path.join(full_d, f"latent{i}.npy") for i in range(1, 4)]
                    if all(os.path.exists(f) for f in files):
                        self.subdirs.append(full_d)
        self.subdirs = sorted(self.subdirs)

    def __len__(self):
        return len(self.subdirs)

    def __getitem__(self, idx):
        subdir = self.subdirs[idx]
        latent1 = np.load(os.path.join(subdir, "latent1.npy"))
        latent2 = np.load(os.path.join(subdir, "latent2.npy"))
        latent3 = np.load(os.path.join(subdir, "latent3.npy"))
        # Convert to tensor (assume float32)
        latent1 = torch.tensor(latent1, dtype=torch.float32)
        latent2 = torch.tensor(latent2, dtype=torch.float32)
        latent3 = torch.tensor(latent3, dtype=torch.float32)
        return latent1, latent2, latent3

def train(args):
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader.
    dataset = LatentDataset(args.data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Select model type from MODEL_DICT.
    if args.model not in MODEL_DICT:
        raise ValueError(f"Model {args.model} not recognized. Available models: {list(MODEL_DICT.keys())}")
    model_class = MODEL_DICT[args.model]
    model = model_class(latent_dim=args.latent_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    early_stopper = EarlyStopper(patience=args.patience, min_delta=1e-4)
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for latent1, latent2, latent3 in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            latent1 = latent1.to(device)
            latent2 = latent2.to(device)
            latent3 = latent3.to(device)
            optimizer.zero_grad()
            # Predict latent2 from latent1 and latent3
            pred = model(latent1, latent3)  # Expect model's forward to accept (latent1, latent3)
            loss = F.mse_loss(pred, latent2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * latent1.size(0)
        train_loss /= len(train_dataset)
        history["train_loss"].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for latent1, latent2, latent3 in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                latent1 = latent1.to(device)
                latent2 = latent2.to(device)
                latent3 = latent3.to(device)
                pred = model(latent1, latent3)
                loss = F.mse_loss(pred, latent2)
                val_loss += loss.item() * latent1.size(0)
        val_loss /= len(val_dataset)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model}_epoch{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch+1, val_loss, checkpoint_path)
        
        # Early stopping
        if early_stopper.should_stop(val_loss):
            print("Early stopping triggered.")
            break

    # Plot metrics and save figure.
    os.makedirs(args.results_dir, exist_ok=True)
    plot_path = os.path.join(args.results_dir, f"{args.model}_loss.png")
    plot_metrics(history, plot_path)
    print(f"Training complete. Metrics plot saved to {plot_path}.")

def evaluate(args):
    # Evaluation command that loads a checkpoint and computes metrics on the test set.
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    dataset = LatentDataset(args.data_dir)
    # Assume 80/10/10 split; here we take last 10% for test.
    test_size = int(0.1 * len(dataset))
    _, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    if args.model not in MODEL_DICT:
        raise ValueError(f"Model {args.model} not recognized. Available models: {list(MODEL_DICT.keys())}")
    model_class = MODEL_DICT[args.model]
    model = model_class(latent_dim=args.latent_dim).to(device)
    
    # Load checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    mse_total = 0.0
    count = 0
    with torch.no_grad():
        for latent1, latent2, latent3 in tqdm(test_loader, desc="Evaluating"):
            latent1 = latent1.to(device)
            latent2 = latent2.to(device)
            latent3 = latent3.to(device)
            pred = model(latent1, latent3)
            mse = F.mse_loss(pred, latent2, reduction='sum').item()
            mse_total += mse
            count += latent1.size(0)
    avg_mse = mse_total / count
    print(f"Test Average MSE: {avg_mse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train latent interpolation models.")
    parser.add_argument("--model", type=str, default="nn", help="Model type. Options: " + ", ".join(MODEL_DICT.keys()))
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gpu_index", type=int, default=3, help="GPU index to use")
    parser.add_argument("--latent_dim", type=int, default=4, help="Latent dimension size")
    parser.add_argument("--data_dir", type=str, default="/data/ggonsior/atd12k", help="Directory with train/val/test latent data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save training charts")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Train or evaluate the model")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (for evaluation)")
    parser.add_argument("--patience", type=int, default=20, help="Number of epochs the training process will continue after loss stops improving")
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.mode == "train":
        train(args)
    else:
        if args.checkpoint == "":
            raise ValueError("Evaluation mode requires --checkpoint to be specified.")
        evaluate(args)
