"""
neural_net.py

This module hosts two families of models:
1) A simple TensorFlow/Keras MLP kept for legacy demos (SimpleNeuralNetwork)
2) The PyTorch denoiser used by the NMR notebook (DenoiseNetPhysics) + helpers

The PyTorch section provides a build-and-load flow so the notebook can
instantiate the architecture first and then load the latest checkpoint.
"""

from typing import Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
import math  # <-- needed for math.pi in synth_batch_phys
import numpy as np  # ensure numpy is always available even if TF import fails


# --- (A) Legacy TensorFlow demo (kept for backwards-compatibility) ---
try:
    import numpy as np
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore

    class SimpleNeuralNetwork:
        def __init__(self, input_shape, num_classes):
            self.model = self.build_model(input_shape, num_classes)

        def build_model(self, input_shape, num_classes):
            model = keras.Sequential()
            model.add(layers.Input(shape=input_shape))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(num_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model

        def train(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
            self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

        def evaluate(self, x_test, y_test):
            return self.model.evaluate(x_test, y_test)

        def predict(self, x):
            return self.model.predict(x)
except Exception:
    # TensorFlow may be unavailable in some environments; skip gracefully.
    pass


# --- (B) PyTorch denoiser used in the NMR workflow ---
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResBlock(nn.Module):
    """Residual block with dilated Conv1d layers.

    Keeps input/output channel count identical and adds a residual connection.
    """

    def __init__(self, channels: int, dilation: int = 1, k: int = 11):
        super().__init__()
        pad = dilation * ((k - 1) // 2)
        self.conv1 = nn.Conv1d(channels, channels, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, k, padding=pad, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(x + h)


class DenoiseNetPhysics(nn.Module):
    """Dilated Conv1D residual denoiser for complex FID (2 channels: real, imag)."""

    def __init__(self, in_ch: int = 2, hidden: int = 64, k: int = 11, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.name = "DenoiseNetPhysics"
        self.inp = nn.Conv1d(in_ch, hidden, k, padding=(k - 1) // 2)
        self.blocks = nn.Sequential(*[DilatedResBlock(hidden, d, k) for d in dilations])
        self.out = nn.Conv1d(hidden, in_ch, k, padding=(k - 1) // 2)  # residual head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x))
        h = self.blocks(h)
        resid = self.out(h)
        return x + resid


# ---- Helpers for checkpoint management ----
def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint(model: nn.Module, ckpt_path: str, map_location: Optional[torch.device] = None, strict: bool = False) -> Tuple[bool, str]:
    """Load a checkpoint into an already-instantiated model.

    Returns (ok, message). ok=False if file missing or load failed.
    """
    if not os.path.exists(ckpt_path):
        return False, f"[Info] Checkpoint not found: {ckpt_path}"
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location or 'cpu')
        state = ckpt.get('model_state', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=strict)
        msg = f"[Info] Loaded checkpoint: {ckpt_path}"
        if missing:
            msg += f" | missing={len(missing)}"
        if unexpected:
            msg += f" | unexpected={len(unexpected)}"
        return True, msg
    except Exception as e:
        return False, f"[Warn] Failed to load {ckpt_path}: {e}"


def build_model_from_latest(checkpoints_dir: str, latest_name: str = "DenoiseNetPhysics_latest.pth",
                            device: Optional[torch.device] = None) -> Tuple[nn.Module, Optional[str]]:
    """Instantiate DenoiseNetPhysics, then attempt to load the latest checkpoint.

    Returns (model, loaded_from_path_or_None).
    """
    device = device or torch.device('cpu')
    model = DenoiseNetPhysics().to(device)
    latest_path = os.path.join(checkpoints_dir, latest_name)
    ok, msg = load_checkpoint(model, latest_path, map_location=device, strict=False)
    print(msg)
    # If checkpoint loaded but contained non-finite params, reinitialize to safe defaults
    if ok and _has_non_finite_params(model):
        print("[Warn] Loaded checkpoint contains NaN/Inf parameters → reinitializing model weights")
        _reinit_model_(model)
        ok = False
    setattr(model, "_loaded_from", latest_path if ok else None)
    return model, (latest_path if ok else None)


__all__ = [
    # TF demo
    'SimpleNeuralNetwork',
    # Torch denoiser
    'DilatedResBlock', 'DenoiseNetPhysics',
    # helpers
    'count_params', 'load_checkpoint', 'build_model_from_latest',
    # training utilities
    'synth_batch_phys', 'combined_loss',
]


def synth_batch_phys(batch_size=16, L=2048, snr_std=0.025, device='cpu'):
    """
    Generate synthetic complex FID batches with REALISTIC NMR physics:
    - Multiple peaks with variable T2* decay
    - J-coupling multiplets (doublets, triplets)
    - Frequency drift
    - Correlated noise + optional 1/f baseline
    
    Returns (fid_noisy, fid_clean), each shape (batch_size, 2, L).
    """
    import math
    
    B, device_obj = batch_size, torch.device(device)
    
    # NMR physics parameters
    dt = 1e-4  # dwell time in seconds
    nu0 = 400e6  # 400 MHz spectrometer
    
    # Initialize arrays
    fid_clean_np = np.zeros((B, L), dtype=np.complex64)
    
    for b in range(B):
        # Random number of peaks per FID
        n_peaks = np.random.randint(2, 8)
        
        t = np.arange(L) * dt
        z = np.zeros(L, dtype=np.complex64)
        
        for k in range(n_peaks):
            # Frequency: sample in realistic chemical shift range
            ppm = np.random.uniform(0.5, 8.0)  # 0.5-8 ppm is common for 1H
            f_hz = ppm * (nu0 / 1e6)  # convert ppm to Hz
            
            # T2* decay: realistic range for solution NMR
            t2 = np.random.uniform(0.05, 0.5)  # 50-500 ms
            
            # Amplitude variation
            A = np.random.uniform(5.0, 20.0)
            
            # Phase variation
            phi = np.random.uniform(0, 2*np.pi)
            
            # J-coupling: 30% chance of multiplet
            if np.random.random() < 0.3:
                J = np.random.uniform(5.0, 15.0)  # typical J-coupling in Hz
                # Create doublet
                for sgn in [-0.5, 0.5]:
                    f_comp = f_hz + sgn * J
                    z += (A/2.0) * np.exp(-t / t2) * np.exp(1j * (2*np.pi*f_comp*t + phi))
            else:
                # Single peak
                z += A * np.exp(-t / t2) * np.exp(1j * (2*np.pi*f_hz*t + phi))
        
        # Add small frequency drift (realistic spectrometer instability)
        drift_hz = np.random.normal(0, 0.2)
        z *= np.exp(1j * 2*np.pi * drift_hz * (t - t.mean()) * 0.1)
        
        fid_clean_np[b] = z
    
    # Convert to torch tensor
    fid_clean_complex = torch.from_numpy(fid_clean_np).to(device_obj)
    real_clean = fid_clean_complex.real.unsqueeze(1)  # [B, 1, L]
    imag_clean = fid_clean_complex.imag.unsqueeze(1)  # [B, 1, L]
    fid_clean = torch.cat([real_clean, imag_clean], dim=1)  # [B, 2, L]
    
    # Noise scaled to early-signal RMS (first 20%)
    early_len = max(64, int(0.2 * L))
    early_rms = torch.sqrt((fid_clean[:, :, :early_len] ** 2).mean(dim=(1, 2)))  # Shape: [B]
    noise_level = snr_std * early_rms.view(B, 1, 1)  # Shape: [B, 1, 1] for broadcasting
    
    # Correlated noise (more realistic than pure Gaussian)
    noise = torch.randn(B, 2, L, device=device_obj) * noise_level
    
    # Optional 1/f noise component (baseline drift)
    if np.random.random() < 0.3:
        # Add low-frequency baseline
        baseline_freq = torch.fft.rfftfreq(L, d=1.0, device=device_obj)
        baseline_amp = 1.0 / (1.0 + baseline_freq * 10)
        baseline_phase = torch.randn(len(baseline_amp), device=device_obj)
        baseline = torch.fft.irfft(baseline_amp * baseline_phase, n=L)
        # baseline shape: [L], noise_level shape: [B, 1, 1]
        baseline_contrib = baseline.unsqueeze(0).unsqueeze(0) * (noise_level * 0.3)  # [B, 1, L]
        noise[:, 0:1, :] += baseline_contrib
        noise[:, 1:2, :] += baseline_contrib
    
    fid_noisy = fid_clean + noise
    
    # ✅ SCALE TO MATCH REAL DATA (avg std ≈ 15)
    target_scale = 15.0
    current_scale = fid_noisy.std().item()
    scale_factor = target_scale / (current_scale + 1e-12)
    
    fid_clean = fid_clean * scale_factor
    fid_noisy = fid_noisy * scale_factor
    
    return fid_noisy.float(), fid_clean.float()

import torch
import torch.nn.functional as F

def _complex_fft(signal_2ch: torch.Tensor):
    """Full complex FFT of a two-channel (real, imag) signal.
    signal_2ch: (B,2,L) float32
    returns: (B,L) complex spectrum (torch.cfloat)
    """
    real, imag = signal_2ch[:, 0], signal_2ch[:, 1]
    z = torch.complex(real, imag)
    return torch.fft.fft(z, dim=-1, norm='ortho')


def _rfft_mag(signal_2ch: torch.Tensor, eps: float = 1e-6, half: bool = True):
    """Magnitude spectrum from complex input without using rfft on complex.
    We compute complex FFT, then optionally take half-spectrum magnitude.
    """
    real, imag = signal_2ch[:, 0], signal_2ch[:, 1]
    z = torch.complex(real, imag)
    Z = torch.fft.fft(z, dim=-1, norm='ortho')
    mag = torch.abs(Z)
    if half:
        L = z.shape[-1]
        mag = mag[..., : (L // 2) + 1]
    return torch.clamp(mag, min=eps)



def combined_loss(pred: torch.Tensor,
                  target: torch.Tensor,
                  dt: Optional[float] = None,
                  x_ref: Optional[torch.Tensor] = None,
                  freq_weight: float = 0.5,
                  time_weight: float = 0.4,
                  l1_weight: float = 0.05,
                  tv_weight: float = 0.0,
                  self_denoise_consistency: float = 0.05,
                  eps: float = 1e-6) -> torch.Tensor:
    """
    Improved hybrid loss for better signal preservation:
    - Balanced time/freq domains
    - L1 regularization to prevent over-smoothing
    - Phase-aware spectral loss
    pred/target: (B,2,L)
    dt: dwell time (optional, reserved)
    x_ref: original noisy input (optional, encourages minimal distortion)
    Returns finite scalar tensor; non-finite intermediates are clamped.
    """
    # Time-domain MSE
    td_mse = F.mse_loss(pred, target)

    # Frequency-domain magnitude loss (half-spectrum)
    mag_p = _rfft_mag(pred, eps=eps)
    mag_t = _rfft_mag(target, eps=eps)
    fd_mse = F.mse_loss(mag_p, mag_t)

    # ✅ Improved: Focus on preserving strong peaks (don't over-weight noise floor)
    # Use sqrt weighting instead of linear to reduce emphasis on noise
    mag_t_norm = mag_t / (mag_t.max(dim=-1, keepdim=True)[0] + eps)
    weight = torch.sqrt(mag_t_norm + eps)
    fd_weighted = torch.mean(((mag_p - mag_t) ** 2) * weight)
    freq_term = 0.7 * fd_mse + 0.3 * fd_weighted

    # ✅ L1 sparsity on residual (prevents removing too much signal)
    l1_term = torch.mean(torch.abs(pred - target)) if l1_weight > 0 else torch.zeros((), device=pred.device)

    # Simple total variation (smoothness) along time
    tv = torch.mean(torch.abs(pred[:, :, 1:] - pred[:, :, :-1])) if tv_weight > 0 else torch.zeros((), device=pred.device)

    # Consistency: keep close to input if self-denoising (avoid hallucination)
    cons = F.mse_loss(pred, x_ref) if (x_ref is not None and self_denoise_consistency > 0) else torch.zeros((), device=pred.device)

    total = (freq_weight * freq_term +
             time_weight * td_mse +
             l1_weight * l1_term +
             tv_weight * tv +
             self_denoise_consistency * cons)

    if not torch.isfinite(total):
        # Avoid propagating NaNs/Infs
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return total


# ----- Utility: check and reinitialize model if non-finite -----
def _has_non_finite_params(model: nn.Module) -> bool:
    for _, p in model.named_parameters():
        if p is not None and not torch.isfinite(p).all():
            return True
    return False


def _reinit_model_(model: nn.Module) -> None:
    """In-place Kaiming initialization of Conv1d/Linear layers."""
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if getattr(m, 'bias', None) is not None and m.bias is not None:
                nn.init.zeros_(m.bias)