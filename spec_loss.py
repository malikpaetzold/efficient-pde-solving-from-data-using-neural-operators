import math
import torch
import torch.nn as nn


class FNOSpectralLoss(nn.Module):
    def __init__(self,
                 n_modes: int = 16,
                 lambda_recon: float = 1.0,
                 lambda_mid:   float = 0.4,
                 lambda_high:  float = 0.6,
                 k_lo: int = 4,
                 k_hi: int | None = None,
                 warmup_epochs: int = 10,
                 ramp_epochs:   int = 45,
                 schedule: str = 'cosine',
                 eps: float = 1e-8):
        super().__init__()
        if k_hi is None:
            k_hi = max(k_lo + 2, (k_lo + n_modes) // 2)
        assert 0 < k_lo < k_hi <= n_modes
        self.n_modes = n_modes
        self.lambda_recon = lambda_recon
        self.lambda_mid_max  = lambda_mid
        self.lambda_high_max = lambda_high
        self.k_lo, self.k_hi = k_lo, k_hi
        self.eps = eps

        self.warmup_epochs = warmup_epochs
        self.ramp_epochs   = ramp_epochs
        self.schedule      = schedule
        self._cache = {}

    def set_epoch(self, epoch: int):
        self.current_epoch.fill_(epoch)

    def _ramp(self, e: int) -> float:
        if self.schedule == 'none':
            return 1.0
        if e < self.warmup_epochs:
            return 0.0
        if e >= self.warmup_epochs + self.ramp_epochs:
            return 1.0
        t = (e - self.warmup_epochs) / max(1, self.ramp_epochs)
        if self.schedule == 'linear':
            return t
        
        # cosine
        return 0.5 * (1.0 - math.cos(math.pi * t))

    def _radii(self, H, W, device):
        key = (H, W, str(device))
        if key not in self._cache:
            ky = torch.fft.fftfreq(H,  d=1.0 / H, device=device)
            kx = torch.fft.rfftfreq(W, d=1.0 / W, device=device)
            kyG, kxG = torch.meshgrid(ky, kx, indexing='ij')
            self._cache[key] = torch.sqrt(kyG**2 + kxG**2).floor().to(torch.long)
        return self._cache[key]

    @staticmethod
    def _rel_l2(num_sq, den_sq, eps):
        return (torch.sqrt(num_sq + eps) / torch.sqrt(den_sq + eps)).mean()

    def forward(self, pred, target, epoch: int):
        err = pred - target

        recon = self._rel_l2(
            torch.sum(err**2, dim=(-2, -1)),
            torch.sum(target**2, dim=(-2, -1)),
            self.eps,
        )

        err_pwr = torch.fft.rfft2(err, norm='ortho').abs() ** 2
        tgt_pwr = torch.fft.rfft2(target, norm='ortho').abs() ** 2

        H, W = pred.shape[-2:]
        radii   = self._radii(H, W, pred.device)
        in_mid  = (radii >= self.k_lo) & (radii <  self.k_hi)
        in_high = (radii >= self.k_hi) & (radii <= self.n_modes)

        mid = torch.sqrt((err_pwr * in_mid).sum() + self.eps) \
            / torch.sqrt((tgt_pwr * in_mid).sum() + self.eps)
        
        high = self._rel_l2(
            (err_pwr * in_high).sum(dim=(-2, -1)),
            (tgt_pwr * in_high).sum(dim=(-2, -1)),
            self.eps,
        )

        ramp = 1.0 if epoch is None else self._ramp(epoch)
        lam_mid  = self.lambda_mid_max  * ramp
        lam_high = self.lambda_high_max * ramp

        total = self.lambda_recon * recon + lam_mid * mid + lam_high * high

        return total, {
            'recon': recon.detach(),
            'mid':   mid.detach(),
            'high':  high.detach(),
            'total': total.detach(),
            'ramp':  torch.tensor(ramp, device=pred.device),
        }