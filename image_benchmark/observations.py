"""Shared operators in pixel units [-1, 1]; never clip inside the likelihood."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SuperResolution(nn.Module):
    """Nonoverlapping block averaging (linear), not bicubic interpolation."""
    def __init__(self, factor=4):
        super().__init__()
        if not isinstance(factor, int) or factor < 1:
            raise ValueError("factor must be a positive integer")
        self.factor = factor

    def forward(self, image):
        if any(n % self.factor for n in image.shape[-2:]):
            raise ValueError("Image dimensions must be divisible by the SR factor")
        return F.avg_pool2d(image, self.factor, self.factor)


class FourierMagnitude(nn.Module):
    """Smoothed magnitude of an orthonormal, centered, zero-padded 2D FFT.

    sqrt(|F x|^2 + epsilon^2) is the EXACT operator used for both data
    generation and inference. A small positive epsilon makes gradients
    defined at zero. Each RGB channel is measured separately.
    """
    def __init__(self, oversample=2.0, epsilon=1e-6):
        super().__init__()
        if oversample < 1 or epsilon <= 0:
            raise ValueError("oversample >= 1 and epsilon > 0 are required")
        self.oversample, self.epsilon = float(oversample), float(epsilon)

    def forward(self, image):
        h, w = image.shape[-2:]
        ph = math.ceil(h * self.oversample) - h
        pw = math.ceil(w * self.oversample) - w
        padded = F.pad(image, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        fourier = torch.fft.fftshift(torch.fft.fft2(padded, norm="ortho"), dim=(-2, -1))
        return (fourier.real.square() + fourier.imag.square() + self.epsilon**2).sqrt()


def make_operator(task, config):
    if task == "super_resolution":
        return SuperResolution(config["factor"])
    if task == "phase_retrieval":
        return FourierMagnitude(config["oversample"], config["epsilon"])
    raise ValueError(f"Unknown task: {task}")


class PixelLikelihood:
    def __init__(self, operator, measurement, sigma):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        self.operator = operator
        self.measurement = measurement.detach()
        self.sigma = float(sigma)

    def residual(self, image):
        return self.operator(image) - self.measurement

    def __call__(self, image):
        # Summation, not MSE: division by the number of pixels changes the target.
        residual = self.residual(image)
        return 0.5 * residual.double().square().flatten(1).sum(1) / self.sigma**2

    @property
    def measurement_dim(self):
        return self.measurement[0].numel()
