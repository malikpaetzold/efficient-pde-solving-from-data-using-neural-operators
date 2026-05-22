import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import xarray
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from models import FNO2d
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.losses import LpLoss, H1Loss
import torch.nn.functional as F
from lossy import HighPassLoss
from radial_binned_spectral_loss import RadialBinnedSpectralLoss
from spec_loss import SpectralLoss

torch.manual_seed(42)
np.random.seed(42)

CONFIG = {
    'data_path': 'nsforcing-64_1e3_25_std-1400.nc',
    'train_split': 0.6,
    'val_split': 0.2,
    'max_samples': 1400,
    'target_resolution': None,
    
    'n_modes': 20,
    'hidden_channels': 40,
    'n_layers': 8,
    'in_channels': 10,
    'out_channels': 1,
    
    'batch_size': 16,
    'epochs': 80,
    'learning_rate': 8e-4,
    'weight_decay': 2e-5,
    'scheduler_step': 50,
    'scheduler_milestones': [50, 70],
    'scheduler_gamma': 0.5,
    
    'loss_type': 'fno_spectral',
    'fno_loss_kwargs': {
        'lambda_recon': 1.0,
        'lambda_mid': 0.1,
        'lambda_high': 0.2,
        'k_lo': 4,
        'k_hi': 12,
    },
    
    'checkpoint_dir': 'checkpoints4',
    'plot_dir': 'plots3',
    'save_every': 10,
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


class NavierStokesDataset(Dataset):
    def __init__(self, data_path, indices, n_input_steps=10, target_resolution=None):
        """
        Args:
            data_path: path to NetCDF file
            indices: sample indices (optional)
            n_input_steps: number of timesteps to use as input
            target_resolution: tuple to resize data to, or none to keep original
        """
        self.ds = xarray.open_dataset(data_path)
        self.indices = indices
        self.n_input_steps = n_input_steps
        self.target_resolution = target_resolution
        
        self.data = self.ds["vorticity"].isel(sample=indices).values
        
        self.n_samples = len(indices)
        self.n_timesteps = self.data.shape[1]
        self.original_resolution = (self.data.shape[2], self.data.shape[3])
        
    def __len__(self):
        return self.n_samples * (self.n_timesteps - self.n_input_steps)
    
    def __getitem__(self, idx):
        sample_idx = idx // (self.n_timesteps - self.n_input_steps)
        time_idx = idx % (self.n_timesteps - self.n_input_steps)
        
        # input: n_input_steps consecutive timesteps
        x = self.data[sample_idx, time_idx:time_idx + self.n_input_steps]
        # output: next timesteps
        y = self.data[sample_idx, time_idx + self.n_input_steps:time_idx + self.n_input_steps + 1]
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        # resize if needed
        if self.target_resolution is not None:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0),
                size=self.target_resolution,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            y = torch.nn.functional.interpolate(
                y.unsqueeze(0),
                size=self.target_resolution,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return x, y



def create_dataloaders(config):
    ds = xarray.open_dataset(config['data_path'])
    num_samples = ds.sizes['sample']
    ds.close()
    
    max_samples = config.get('max_samples', num_samples)
    num_samples = min(max_samples, num_samples)
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_end = int(config['train_split'] * num_samples)
    val_end = train_end + int(config['val_split'] * num_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    print(f"Dataset splits: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    
    train_dataset = NavierStokesDataset(
        config['data_path'], 
        train_indices, 
        n_input_steps=config['in_channels'],
        target_resolution=config['target_resolution']
    )
    val_dataset = NavierStokesDataset(
        config['data_path'], 
        val_indices, 
        n_input_steps=config['in_channels'],
        target_resolution=config['target_resolution']
    )
    test_dataset = NavierStokesDataset(
        config['data_path'], 
        test_indices, 
        n_input_steps=config['in_channels'],
        target_resolution=config['target_resolution']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        generator=generator,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        
        if torch.isnan(loss):
            print(f"NaN detected at batch {batch_idx}, skipping...")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item()
            
            mse = F.mse_loss(pred, y, reduction='sum')
            mae = F.l1_loss(pred, y, reduction='sum')
            total_mse += mse.item()
            total_mae += mae.item()
            num_samples += y.numel()
    
    val_mse = total_mse / num_samples
    val_mae = total_mae / num_samples
    return val_loss / len(val_loader), val_mse, val_mae

def plot_predictions(model, val_loader, device, save_path, num_samples=4):
    model.eval()
    
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        y_pred = model(x)
    
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    num_samples = min(num_samples, x.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        im0 = axes[i, 0].imshow(x[i, -1], cmap='RdBu_r', aspect='auto')
        axes[i, 0].set_title(f'Input (t={x.shape[1]-1})', fontsize=10)
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        im1 = axes[i, 1].imshow(y[i, 0], cmap='RdBu_r', aspect='auto')
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        im2 = axes[i, 2].imshow(y_pred[i, 0], cmap='RdBu_r', aspect='auto')
        axes[i, 2].set_title('Prediction', fontsize=10)
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        error = np.abs(y[i, 0] - y_pred[i, 0]).mean()
        axes[i, 2].text(0.5, -0.15, f'MAE: {error:.6f}', 
                       transform=axes[i, 2].transAxes,
                       ha='center', fontsize=9)
    
    plt.suptitle('Validation Predictions', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction visualization saved to {save_path}")


def main():
    print("start time: ", datetime.now())
    Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)
    Path(CONFIG['plot_dir']).mkdir(exist_ok=True)
    
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(CONFIG)
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    print("Creating FNO model...")
    model = FNO2d(
        n_modes=CONFIG['n_modes'],
        in_channels=CONFIG['in_channels'],
        out_channels=CONFIG['out_channels'],
        hidden_channels=CONFIG['hidden_channels'],
        n_layers=CONFIG['n_layers'],
    ).to(device)

    metrics_file = os.path.join(CONFIG["checkpoint_dir"], "metrics.csv")
    with open(metrics_file, "w") as f:
        f.write("epoch,val_loss,val_mse,val_mae\n")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=1e-5
    )
    
    # Create loss function
    if CONFIG['loss_type'] == 'l2':
        loss_fn = LpLoss(d=2, p=2)
    elif CONFIG['loss_type'] == 'rbsl':
        mse_loss = LpLoss(d=2, p=2)
        def loss_fn(pred, target):
            freq_losses = RadialBinnedSpectralLoss(pred, target, reduction="mean")
            return (freq_losses.mean()) + mse_loss(pred, target)
    elif CONFIG['loss_type'] == 'fno_spectral':
        from rbsl2 import FNOSpectralLoss
        _loss_module = FNOSpectralLoss(
            n_modes=CONFIG['n_modes'],
            lambda_recon=1.0,
            lambda_mid=0.1,
            lambda_high=0.2,
            k_lo=4,
        ).to(device)

        def loss_fn(pred, target):
            loss, _components = _loss_module(pred, target)
            return loss
    else:
        loss_fn.components_module = _loss_module
        raise ValueError(f"Unknown loss type: {CONFIG['loss_type']}")
    
    print(f"\nStarting training for {CONFIG['epochs']} epochs...")
    train_losses = []
    val_losses = []
    best_val_mse = float('inf')
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        if hasattr(loss_fn, 'components_module'):
            loss_fn.components_module.set_epoch(epoch - 1)
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        train_losses.append(train_loss)

        val_loss, val_mse, val_mae = validate(model, val_loader, loss_fn, device, epoch)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"Epoch {epoch}/{CONFIG['epochs']} | train {train_loss:.6f} | "
            f"val_loss {val_loss:.6f} | val_mse {val_mse:.6f} | val_mae {val_mae:.6f}")
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mse': val_mse,
                'val_mae': val_mae,
                'config': CONFIG,
            }, Path(CONFIG['checkpoint_dir']) / 'best_model.pt')
            print(f"  → new best (val_mse={val_mse:.6f})")

        with open(metrics_file, "a") as f:
            f.write(f"{epoch},{val_loss:.6f},{val_mse:.6f},{val_mae:.6f}\n")
        
        if epoch % CONFIG['save_every'] == 0:
            checkpoint_path = Path(CONFIG['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': CONFIG
            }, checkpoint_path)

            plot_path = Path(CONFIG['plot_dir']) / f'training_history_epoch_{epoch}.png'
            
            pred_plot_path = Path(CONFIG['plot_dir']) / f'predictions_epoch_{epoch}.png'
            plot_predictions(model, val_loader, device, pred_plot_path, num_samples=4)
    
    print("\nEvaluating on test set...")
    test_loss, test_mse, test_mae = validate(model, test_loader, loss_fn, device, epoch)
    print(f"Test Loss: {test_loss:.6f} | Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")
    
    final_checkpoint_path = Path(CONFIG['checkpoint_dir']) / 'final_model.pt'
    torch.save({
        'epoch': CONFIG['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
        'config': CONFIG
    }, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    
    plot_path = Path(CONFIG['plot_dir']) / 'training_history.png'
    
    test_pred_plot_path = Path(CONFIG['plot_dir']) / 'predictions_test_final.png'
    plot_predictions(model, test_loader, device, test_pred_plot_path, num_samples=4)
    
    print("\nTraining complete!")


def load_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = FNO2d(
        n_modes=config['n_modes'],
        hidden_channels=config['hidden_channels'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        n_layers=config['n_layers']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    if 'test_loss' in checkpoint:
        print(f"Test loss: {checkpoint['test_loss']:.6f}")
    
    return model, config


if __name__ == '__main__':
    main()