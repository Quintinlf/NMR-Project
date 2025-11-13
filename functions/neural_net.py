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


def synth_batch_phys(batch_size: int, L: int, snr_std: float = 0.03, colored_noise: bool = True, device: Optional[torch.device] = None):
    """
    Generate synthetic complex FIDs: sum of damped complex sinusoids + optional colored noise.
    Returns (x_noisy, y_clean) with shape (B, 2, L), float32, on CUDA if available.
    Noise scaling is robust to vanishing tails by referencing the 20th-percentile envelope.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = int(batch_size)
    n = torch.arange(L, device=device, dtype=torch.float32)[None, :]  # (1,L)

    y_real = torch.zeros(B, L, device=device, dtype=torch.float32)
    y_imag = torch.zeros(B, L, device=device, dtype=torch.float32)
    two_pi = 2.0 * math.pi

    Kmin, Kmax = 2, 6
    for b in range(B):
        # sample number of components per sample without relying on numpy
        K = int(torch.randint(Kmin, Kmax + 1, (1,), device=device))
        f = torch.empty(K, device=device).uniform_(-0.45, 0.45)             # cycles/sample in Nyquist
        alpha = torch.empty(K, device=device).uniform_(1e-3, 8e-3)          # per-sample decay
        phi = torch.empty(K, device=device).uniform_(0, 2 * math.pi)
        A = (torch.rand(K, device=device) ** 2)                             # amplitudes

        env = torch.exp(-alpha[:, None] * n)                                # (K,L)
        ang = two_pi * f[:, None] * n + phi[:, None]                        # (K,L)
        y_real[b] = (A[:, None] * env * torch.cos(ang)).sum(dim=0)
        y_imag[b] = (A[:, None] * env * torch.sin(ang)).sum(dim=0)

    # Optional slow drift
    if colored_noise:
        t = torch.linspace(-1, 1, L, device=device)
        y_real = y_real + (0.02 * torch.randn(B, 1, device=device)) * (t**2)[None, :]
        y_imag = y_imag + (0.02 * torch.randn(B, 1, device=device)) * (t**2)[None, :]

    # Robust amplitude reference from envelope (avoid tiny tails)
    env = torch.sqrt(y_real**2 + y_imag**2)  # (B,L)
    # 20th percentile per batch element
    q20 = torch.quantile(env, 0.20, dim=1).clamp_min(1e-3)  # (B,)
    sigma = (snr_std * q20).clamp(min=1e-6)                 # (B,)
    nr = torch.randn(B, L, device=device) * sigma[:, None]
    ni = torch.randn(B, L, device=device) * sigma[:, None]

    # Simple coloring (exponential kernel)
    if colored_noise:
        klen = max(5, int(0.03 * L))
        kern = torch.exp(-torch.linspace(0, 4.0, klen, device=device))
        kern = kern / kern.sum()
        def filt(x):
            x1 = x.unsqueeze(1)
            pad = klen - 1
            xpad = torch.nn.functional.pad(x1, (pad, 0), mode="reflect")
            return torch.nn.functional.conv1d(xpad, kern.view(1, 1, -1)).squeeze(1)
        nr = filt(nr); ni = filt(ni)

    x_real = y_real + nr
    x_imag = y_imag + ni

    y = torch.stack([y_real, y_imag], dim=1).to(torch.float32)  # (B,2,L)
    x = torch.stack([x_real, x_imag], dim=1).to(torch.float32)  # (B,2,L)
    # sanity checks
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise RuntimeError("Non-finite values produced in synth_batch_phys")
    if torch.mean((x - y)**2).item() <= 1e-12:
        raise RuntimeError("synth_batch_phys produced x≈y (noise not added)")
    return x, y

# ...existing code...
import torch
import torch.nn.functional as F
# ...existing code...

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
                  freq_weight: float = 0.6,
                  time_weight: float = 0.4,
                  l1_weight: float = 0.0,
                  tv_weight: float = 0.0,
                  self_denoise_consistency: float = 0.05,
                  eps: float = 1e-6) -> torch.Tensor:
    """
    Stable hybrid loss combining time-domain MSE and rFFT magnitude MSE.
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

    # Amplitude-weighted emphasis on strong spectral bins
    denom = torch.clamp(mag_t.mean(dim=-1, keepdim=True), min=eps)
    weight = mag_t / denom
    fd_weighted = torch.mean(((mag_p - mag_t) ** 2) * weight)
    freq_term = 0.5 * (fd_mse + fd_weighted)

    # L1 sparsity on residual (pred - target)
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
# ...existing code...