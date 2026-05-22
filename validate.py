import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import xarray
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
import json
from scipy.fft import fft2, fftshift

from models import FNO2d

torch.manual_seed(42)
np.random.seed(42)


class NavierStokesDataset(Dataset):
    # from train
    def __init__(self, data_path, indices, n_input_steps=10, target_resolution=None):
        self.ds = xarray.open_dataset(data_path)
        self.indices = indices
        self.n_input_steps = n_input_steps
        self.target_resolution = target_resolution
        
        self.data = self.ds["vorticity"].isel(sample=indices).values
        
        self.n_samples = len(indices)
        self.n_timesteps = self.data.shape[1]
        self.original_resolution = (self.data.shape[2], self.data.shape[3])
        
        if target_resolution is not None:
            print(f"Will resize from {self.original_resolution} to {target_resolution}")
        
    def __len__(self):
        return self.n_samples * (self.n_timesteps - self.n_input_steps)
    
    def __getitem__(self, idx):
        sample_idx = idx // (self.n_timesteps - self.n_input_steps)
        time_idx = idx % (self.n_timesteps - self.n_input_steps)
        
        x = self.data[sample_idx, time_idx:time_idx + self.n_input_steps]
        y = self.data[sample_idx, time_idx + self.n_input_steps:time_idx + self.n_input_steps + 1]
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
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


def compute_2d_fft(field):
    fft_field = fft2(field)
    fft_field = fftshift(fft_field)
    return fft_field


def compute_wavenumbers(shape):
    H, W = shape
    center_y, center_x = H // 2, W // 2
    
    y_coords = np.arange(H) - center_y
    x_coords = np.arange(W) - center_x
    
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    wavenumbers = np.sqrt(X**2 + Y**2)
    
    return wavenumbers


def compute_radial_spectrum(field, n_bins=None):
    H, W = field.shape
    if n_bins is None:
        n_bins = min(H, W) // 2
    
    fft_field = compute_2d_fft(field)
    
    energy_2d = np.abs(fft_field) ** 2
    
    k = compute_wavenumbers(field.shape)
    
    max_k = np.sqrt((H//2)**2 + (W//2)**2)
    bin_edges = np.linspace(0, max_k, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    spectrum = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (k >= bin_edges[i]) & (k < bin_edges[i+1])
        spectrum[i] = np.sum(energy_2d[mask])
    
    return bin_centers, spectrum


def compute_radial_spectrum_batch(fields, n_bins=None):
    spectra = []
    for field in fields:
        k, spec = compute_radial_spectrum(field, n_bins)
        spectra.append(spec)
    
    spectra = np.array(spectra)
    avg_spectrum = np.mean(spectra, axis=0)
    
    return k, avg_spectrum


# ============================================================================
# METRICS
# ============================================================================
class MetricsCalculator:
    @staticmethod
    def l2_error(pred, target):
        """L2 error."""
        return np.sqrt(np.sum((pred - target) ** 2))
    
    @staticmethod
    def relative_l2_error(pred, target):
        """Relative L2 error"""
        return np.sqrt(np.sum((pred - target) ** 2)) / np.sqrt(np.sum(target ** 2))
    
    @staticmethod
    def mae(pred, target):
        """Mean Absolute Error"""
        return np.mean(np.abs(pred - target))
    
    @staticmethod
    def mse(pred, target):
        """Mean Squared Error"""
        return np.mean((pred - target) ** 2)
    
    @staticmethod
    def rmse(pred, target):
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((pred - target) ** 2))
    
    @staticmethod
    def max_error(pred, target):
        """Maximum absolute error"""
        return np.max(np.abs(pred - target))
    
    @staticmethod
    def relative_mae(pred, target):
        """Relative Mean Absolute Error"""
        return np.mean(np.abs(pred - target)) / np.mean(np.abs(target))
    
    @staticmethod
    def compute_all_metrics(pred, target):
        return {
            'l2_error': float(MetricsCalculator.l2_error(pred, target)),
            'relative_l2_error': float(MetricsCalculator.relative_l2_error(pred, target)),
            'mae': float(MetricsCalculator.mae(pred, target)),
            'mse': float(MetricsCalculator.mse(pred, target)),
            'rmse': float(MetricsCalculator.rmse(pred, target)),
            'max_error': float(MetricsCalculator.max_error(pred, target)),
            'relative_mae': float(MetricsCalculator.relative_mae(pred, target)),
        }


class FNOValidator:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict_batch(self, inputs):
        with torch.no_grad():
            inputs = inputs.to(self.device)
            predictions = self.model(inputs)
        return predictions.cpu()
    
    def validate_dataset(
        self, 
        dataloader, 
        output_dir,
        dataset_name='test',
        num_vis_samples=8,
        n_spectral_bins=32
    ):        
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        print("  Running inference...")
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc='Inference')):
            predictions = self.predict_batch(inputs)
            
            all_predictions.append(predictions.numpy())
            all_targets.append(targets.numpy())
            
            if batch_idx * inputs.shape[0] < num_vis_samples:
                all_inputs.append(inputs.numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        if all_inputs:
            inputs = np.concatenate(all_inputs, axis=0)[:num_vis_samples]
        else:
            inputs = None
        
        predictions = predictions[:, 0]
        targets = targets[:, 0]
        
        print(f"  Total samples: {len(predictions)}")
        print(f"  Resolution: {predictions.shape[1]}×{predictions.shape[2]}")
        
        metrics = MetricsCalculator.compute_all_metrics(predictions, targets)
        
        print("\n  Metrics:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.6f}")
        
        metrics_file = Path(output_dir) / f'{dataset_name}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved metrics to {metrics_file}")
        

        output_dir = Path(output_dir)
        
        print("\n  Creating visualizations...")
        if inputs is not None:
            vis_path = output_dir / f'{dataset_name}_field_comparison.png'
            self.visualize_fields(
                inputs[:num_vis_samples],
                targets[:num_vis_samples],
                predictions[:num_vis_samples],
                vis_path
            )
        else:
            print("  Skipping field visualization (no inputs stored)")
        
        # radial spectrum analysis
        spectrum_path = output_dir / f'{dataset_name}_radial_spectrum.png'
        self.visualize_radial_spectrum(
            targets,
            predictions,
            spectrum_path,
            n_bins=n_spectral_bins
        )
        
        # 2D Fourier spectrum visualization
        fft_2d_path = output_dir / f'{dataset_name}_2d_spectrum.png'
        self.visualize_2d_spectrum(
            targets[:num_vis_samples],
            predictions[:num_vis_samples],
            fft_2d_path
        )
        
        return metrics
    
    def visualize_fields(self, inputs, targets, predictions, save_path, num_samples=None):
        if num_samples is None:
            num_samples = len(targets)
        num_samples = min(num_samples, len(targets))
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 3.5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            im0 = axes[i, 0].imshow(inputs[i, -1], cmap='RdBu_r', aspect='auto')
            axes[i, 0].set_title(f'Sample {i+1}\nInput (t={inputs.shape[1]-1})', fontsize=11)
            axes[i, 0].axis('off')
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            im1 = axes[i, 1].imshow(targets[i], cmap='RdBu_r', aspect='auto')
            axes[i, 1].set_title('Ground Truth', fontsize=11)
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            im2 = axes[i, 2].imshow(predictions[i], cmap='RdBu_r', aspect='auto')
            axes[i, 2].set_title('Prediction', fontsize=11)
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            error = np.abs(targets[i] - predictions[i])
            im3 = axes[i, 3].imshow(error, cmap='hot', aspect='auto')
            axes[i, 3].set_title('Absolute Error', fontsize=11)
            axes[i, 3].axis('off')
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
            
            mae = np.abs(targets[i] - predictions[i]).mean()
            rel_l2 = MetricsCalculator.relative_l2_error(predictions[i], targets[i])
            axes[i, 3].text(0.5, -0.15, f'MAE: {mae:.6f} | Rel L2: {rel_l2:.6f}', 
                           transform=axes[i, 3].transAxes,
                           ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def visualize_radial_spectrum(self, targets, predictions, save_path, n_bins=32):
        k_target, spectrum_target = compute_radial_spectrum_batch(targets, n_bins)
        k_pred, spectrum_pred = compute_radial_spectrum_batch(predictions, n_bins)
        
        errors = targets - predictions
        k_error, spectrum_error = compute_radial_spectrum_batch(errors, n_bins)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax = axes[0]
        ax.loglog(k_target, spectrum_target, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.loglog(k_pred, spectrum_pred, 'r--', linewidth=2, label='Prediction', alpha=0.8)
        ax.loglog(k_error, spectrum_error, 'g:', linewidth=2, label='Error', alpha=0.8)
        ax.set_xlabel('Wavenumber k', fontsize=12)
        ax.set_ylabel('Energy E(k)', fontsize=12)
        ax.set_title('Radial Energy Spectrum (Log-Log)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        
        ax = axes[1]
        relative_error = 100 * np.abs(spectrum_pred - spectrum_target) / (spectrum_target + 1e-10)
        ax.semilogx(k_target, relative_error, 'purple', linewidth=2, marker='o', 
                    markersize=4, alpha=0.7)
        ax.set_xlabel('Wavenumber k', fontsize=12)
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_title('Spectral Relative Error vs Wavenumber', fontsize=13, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def visualize_2d_spectrum(self, targets, predictions, save_path, num_samples=None):
        if num_samples is None:
            num_samples = min(4, len(targets))
        num_samples = min(num_samples, len(targets))
        
        fig = plt.figure(figsize=(16, 3.5 * num_samples))
        gs = GridSpec(num_samples, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        for i in range(num_samples):
            fft_target = compute_2d_fft(targets[i])
            fft_pred = compute_2d_fft(predictions[i])
            fft_error = fft_pred - fft_target
            
            log_fft_target = np.log10(np.abs(fft_target) + 1e-10)
            log_fft_pred = np.log10(np.abs(fft_pred) + 1e-10)
            log_fft_error = np.log10(np.abs(fft_error) + 1e-10)
            
            vmin = min(log_fft_pred.min(), log_fft_target.min())
            vmax = max(log_fft_pred.max(), log_fft_target.max())
            
            ax = fig.add_subplot(gs[i, 0])
            im = ax.imshow(targets[i], cmap='RdBu_r', aspect='auto')
            ax.set_title(f'Sample {i+1}\nFlow Field', fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax = fig.add_subplot(gs[i, 1])
            im = ax.imshow(log_fft_target, cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title('Ground Truth\nSpectrum (log)', fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax = fig.add_subplot(gs[i, 2])
            im = ax.imshow(log_fft_pred, cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title('Prediction\nSpectrum (log)', fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax = fig.add_subplot(gs[i, 3])
            im = ax.imshow(log_fft_error, cmap='hot', aspect='auto')
            ax.set_title('Error\nSpectrum (log)', fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('2D Fourier Spectrum Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved 2D spectra visualization to {save_path}")
    
    def visualize_error_distribution(self, targets, predictions, save_path):
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        ax = axes[0, 0]
        ax.hist(errors.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.hist(abs_errors.flatten(), bins=100, alpha=0.7, color='orange', 
                edgecolor='black', log=True)
        ax.set_xlabel('Absolute Error', fontsize=11)
        ax.set_ylabel('Frequency (log scale)', fontsize=11)
        ax.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        sample_mae = np.mean(abs_errors, axis=(1, 2))
        sample_rmse = np.sqrt(np.mean(errors**2, axis=(1, 2)))
        
        x_samples = np.arange(len(sample_mae))
        ax.plot(x_samples, sample_mae, 'o-', label='MAE', alpha=0.7, markersize=3)
        ax.plot(x_samples, sample_rmse, 's-', label='RMSE', alpha=0.7, markersize=3)
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Error', fontsize=11)
        ax.set_title('Per-Sample Error Metrics', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
Error Statistics:
━━━━━━━━━━━━━━━━━━━━
Mean Error:        {errors.mean():.6f}
Std Error:         {errors.std():.6f}
MAE:               {abs_errors.mean():.6f}
RMSE:              {np.sqrt((errors**2).mean()):.6f}
Max Error:         {abs_errors.max():.6f}

Percentiles:
  25th:            {np.percentile(abs_errors, 25):.6f}
  50th (median):   {np.percentile(abs_errors, 50):.6f}
  75th:            {np.percentile(abs_errors, 75):.6f}
  95th:            {np.percentile(abs_errors, 95):.6f}
  99th:            {np.percentile(abs_errors, 99):.6f}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Error Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()


def load_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = {'data_path': 'nsforcing-64_1e3_25_std-1400.nc',
              'train_split': 0.8,
              'val_split': 0.1,
              'target_resolution': None,
              'n_modes': 16,
              'hidden_channels': 60,
              'n_layers': 8,
              'in_channels': 10,
              'out_channels': 1,
              'batch_size': 16,
              'epochs': 80,
              'learning_rate': 0.001,
              'weight_decay': 0.0001,
              'checkpoint_dir': 'gridsearch_spectral/checkpoints',
              'device': 'cuda'}
    config = checkpoint['config']
    
    model = FNO2d(
        n_modes=config['n_modes'],
        hidden_channels=config['hidden_channels'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        n_layers=config['n_layers'],
        coord_features=config.get('coord_features', True),
        activation=config.get('activation', 'gelu'),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    if 'test_loss' in checkpoint:
        print(f"Test loss: {checkpoint['test_loss']:.6f}")
    
    return model, config


def create_dataloader(data_path, indices, config, batch_size=16, num_workers=4):
    dataset = NavierStokesDataset(
        data_path,
        indices,
        n_input_steps=config['in_channels'],
        target_resolution=config.get('target_resolution')
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    return dataloader


def main():
    CONFIG = {
        'checkpoint_path': 'checkpoints4/best_model.pt',
        'output_dir': 'validation_results4',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 16,
        'num_vis_samples': 8,
        'n_spectral_bins': 32,
        
        'datasets': [
            {
                'name': 'test_64',
                'data_path': 'nsforcing-64_1e3_25_std-1400.nc',
                'indices': None, # none uses the same indice split from training
                'description': 'test set from training data (64x64)',
                'batch_size': 16,
            },
            {
                'name': 'test_128',
                'data_path': 'nsforcing-128_1e3_25_std-1400.nc',
                'indices': None,
                'description': 'test at 128x128 resolution',
                'batch_size': 16,
            },
            {
                'name': 'test_256',
                'data_path': 'nsforcing-256_1e3_25_std-1400.nc',
                'indices': None,
                'description': 'test at 256x256 resolution',
                'batch_size': 8,
            },
            {
                'name': 'test_512',
                'data_path': 'nsforcing-512_1e3_25_std-200.nc',
                'indices': range(120),
                'description': 'Generalization test at 512x512 resolution',
                'batch_size': 4,
            },
            {
                'name': 'test_1024',
                'data_path': 'nsforcing-1024_1e3_25_std-40.nc',
                'indices': range(40),
                'description': 'Generalization test at 1024x1024 resolution',
                'batch_size': 1,
            },
            {
                'name': 'test_2048',
                'data_path': 'nsforcing-2048_1e3_25_std-20.nc',
                'indices': range(20),
                'description': 'Generalization test at 2048x2048 resolution',
                'batch_size': 1,
            }
        ]
    }
    
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    print(f"\nLoading model from: {CONFIG['checkpoint_path']}")
    model, train_config = load_model(CONFIG['checkpoint_path'], device)
    print(f"Model architecture:")
    print(model)
    
    validator = FNOValidator(model, device)
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = output_dir / 'validation_config.json'
    with open(config_file, 'w') as f:
        save_config = {k: v for k, v in CONFIG.items() if k != 'datasets'}
        save_config['datasets'] = [
            {k: v for k, v in ds.items() if k != 'indices'}
            for ds in CONFIG['datasets']
        ]
        save_config['model_config'] = train_config
        json.dump(save_config, f, indent=2)
    print(f"Saved configuration to {config_file}")

    all_results = {}
    
    for dataset_config in CONFIG['datasets']:
        dataset_name = dataset_config['name']
        data_path = dataset_config['data_path']
        indices = dataset_config['indices']
        
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Description: {dataset_config.get('description', 'N/A')}")
        print(f"Data path: {data_path}")
        print(f"{'='*70}")
        
        if not Path(data_path).exists():
            print(f"  ⚠️  File not found, skipping: {data_path}")
            continue
        
        if indices is None:
            ds = xarray.open_dataset(data_path)
            num_samples = ds.sizes['sample']
            ds.close()
            
            all_indices = np.arange(num_samples)
            np.random.seed(42)
            np.random.shuffle(all_indices)
            
            train_end = int(train_config.get('train_split', 0.8) * num_samples)
            val_end = train_end + int(train_config.get('val_split', 0.1) * num_samples)
            indices = all_indices[val_end:]
            
            print(f"Using test indices: {len(indices)} samples")
        else:
            print(f"Using specified indices: {len(indices)} samples")
        
        dataset_batch_size = dataset_config.get('batch_size', CONFIG['batch_size'])
        dataset_num_workers = 0 if dataset_batch_size <= 2 else 2
        
        dataloader = create_dataloader(
            data_path, 
            indices, 
            train_config, 
            batch_size=dataset_batch_size,
            num_workers=dataset_num_workers
        )
        
        print(f"Batch size: {dataset_batch_size}, Workers: {dataset_num_workers}")
        
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(exist_ok=True)
        
        metrics = validator.validate_dataset(
            dataloader,
            dataset_output_dir,
            dataset_name=dataset_name,
            num_vis_samples=CONFIG['num_vis_samples'],
            n_spectral_bins=CONFIG['n_spectral_bins']
        )
        
        all_results[dataset_name] = metrics
    
    combined_results_file = output_dir / 'all_results_summary.json'
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'MAE':<12} {'RMSE':<12} {'Rel L2':<12}")
    print(f"{'-'*70}")
    for dataset_name, metrics in all_results.items():
        print(f"{dataset_name:<20} "
                f"{metrics['mae']:<12.6f} "
                f"{metrics['rmse']:<12.6f} "
                f"{metrics['relative_l2_error']:<12.6f}")
    print(f"{'='*70}")
    
    print(f"All results saved to: {output_dir}")
    print(f"Summary file: {combined_results_file}")


if __name__ == '__main__':
    main()