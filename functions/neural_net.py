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
    setattr(model, "_loaded_from", latest_path if ok else None)
    return model, (latest_path if ok else None)


__all__ = [
    # TF demo
    'SimpleNeuralNetwork',
    # Torch denoiser
    'DilatedResBlock', 'DenoiseNetPhysics',
    # helpers
    'count_params', 'load_checkpoint', 'build_model_from_latest',
]


def synth_batch_phys(batch_size: int, L: int, snr_std: float = 0.03, colored_noise: bool = True):
    """
    Generate synthetic complex FIDs: sum of damped complex sinusoids + optional colored noise.
    Returns (x_noisy, y_clean) with shape (B, 2, L), float32, on CUDA if available.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = int(batch_size)
    n = torch.arange(L, device=device, dtype=torch.float32)[None, :]  # (1,L)

    y_real = torch.zeros(B, L, device=device, dtype=torch.float32)
    y_imag = torch.zeros(B, L, device=device, dtype=torch.float32)
    two_pi = 2.0 * math.pi

    Kmin, Kmax = 2, 6
    for b in range(B):
        K = np.random.randint(Kmin, Kmax + 1)
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

    # Noise scaled to tail RMS
    tail = slice(int(0.8 * L), L)
    tail_rms = torch.sqrt((y_real[:, tail]**2 + y_imag[:, tail]**2).mean(dim=1) + 1e-12)  # (B,)
    sigma = (snr_std * tail_rms).clamp(min=1e-6)                                          # (B,)
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
    return x, y

# ...existing code...
import torch
import torch.nn.functional as F
# ...existing code...

def _complex_rfft(signal_2ch: torch.Tensor):
    """
    signal_2ch: (B,2,L) real+imag
    Returns complex spectrum (B,Lf) torch.cfloat
    """
    real, imag = signal_2ch[:,0], signal_2ch[:,1]
    z = torch.complex(real, imag)
    Z = torch.fft.rfft(z, dim=-1, norm='ortho')
    return Z

# ...existing code...
def _complex_fft(signal_2ch: torch.Tensor):
    """
    signal_2ch: (B,2,L) real & imag channels
    returns complex full-spectrum (B,L)
    """
    real = signal_2ch[:,0]
    imag = signal_2ch[:,1]
    z = torch.complex(real, imag)
    return torch.fft.fft(z, dim=-1, norm='ortho')

# replace old _complex_rfft calls inside combined_loss:
# Zp = _complex_rfft(pred)
# Zt = _complex_rfft(target)
# with:
Zp = _complex_fft(pred)
Zt = _complex_fft(target)
# ...existing code...

def combined_loss(pred: torch.Tensor,
                  target: torch.Tensor,
                  dt: float = None,
                  x_ref: torch.Tensor = None,
                  freq_weight: float = 0.6,
                  time_weight: float = 0.4,
                  l1_weight: float = 0.0,
                  tv_weight: float = 0.0,
                  self_denoise_consistency: float = 0.05):
    """
    pred/target: (B,2,L)
    dt: dwell time (optional, for potential future phys terms)
    x_ref: original noisy input (optional, encourages minimal distortion)
    """
    # Time-domain MSE
    td_mse = F.mse_loss(pred, target)

    # Frequency-domain MSE (magnitude)
    Zp = _complex_rfft(pred)
    Zt = _complex_rfft(target)
    mag_p = torch.abs(Zp)
    mag_t = torch.abs(Zt)
    fd_mse = F.mse_loss(mag_p, mag_t)

    # Optional amplitude-weighted emphasis on strong lines
    # (stabilize by adding small floor)
    weight = (mag_t + 1e-6) / (mag_t.mean(dim=-1, keepdim=True) + 1e-6)
    fd_weighted = torch.mean(((mag_p - mag_t) ** 2) * weight)

    freq_term = 0.5 * (fd_mse + fd_weighted)

    # L1 sparsity on residual (pred - target)
    l1_term = torch.mean(torch.abs(pred - target)) if l1_weight > 0 else 0.0

    # Simple total variation (smoothness) along time
    if tv_weight > 0:
        tv = torch.mean(torch.abs(pred[:,:,1:] - pred[:,:,:-1]))
    else:
        tv = 0.0

    # Consistency: keep close to input if self-denoising (avoid hallucination)
    if x_ref is not None and self_denoise_consistency > 0:
        cons = F.mse_loss(pred, x_ref)
    else:
        cons = 0.0

    total = (freq_weight * freq_term +
             time_weight * td_mse +
             l1_weight * l1_term +
             tv_weight * tv +
             self_denoise_consistency * cons)

    return total
# ...existing code...