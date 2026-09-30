"""Controlled curved source-space problems and reference samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import torch
except ImportError:  # The analytic NumPy benchmark does not require PyTorch.
    torch = None

if TYPE_CHECKING:
    from spt_score import SamplerConfig


SCENARIOS = ("banana", "sine")
DEFAULT_BANANA_TAU = 0.3
DEFAULT_SINE_TAU = 0.15
GEOMETRY_VERSION = "full_blocks_v1"


class CurvedObservationProblem:
    """Fixed full-block prior with observations on a configurable leading prefix."""

    def __init__(
        self,
        scenario: str,
        cfg: SamplerConfig,
        sigma_y: float,
        observation_values: np.ndarray,
        observation_tilt: float = 0.0,
        banana_tau: float = DEFAULT_BANANA_TAU,
        sine_tau: float = DEFAULT_SINE_TAU,
        *, observed_blocks: int | None = None,
    ):
        if scenario not in SCENARIOS:
            raise ValueError(scenario)
        if not isinstance(cfg.dim, (int, np.integer)) or cfg.dim < 2 or cfg.dim % 2:
            raise ValueError("dim must be a positive even integer (d = 2 * num_blocks)")
        num_blocks = cfg.dim // 2
        observed_blocks = num_blocks if observed_blocks is None else observed_blocks
        if not isinstance(observed_blocks, (int, np.integer)) or not 1 <= observed_blocks <= num_blocks:
            raise ValueError("observed_blocks must lie in [1, dim/2]")
        if not np.isfinite(sigma_y) or sigma_y <= 0:
            raise ValueError("sigma_y must be positive")
        if not np.isfinite(observation_tilt):
            raise ValueError("observation_tilt must be finite")
        if not np.isfinite(banana_tau) or banana_tau <= 0:
            raise ValueError("banana_tau must be positive and finite")
        if not np.isfinite(sine_tau) or sine_tau <= 0:
            raise ValueError("sine_tau must be positive and finite")
        observation_values = np.asarray(observation_values, dtype=float).reshape(-1)
        if observation_values.size != observed_blocks or not np.all(np.isfinite(observation_values)):
            raise ValueError("observation_values must contain exactly observed_blocks finite entries")
        self.scenario = scenario
        self.cfg = cfg
        self.num_blocks = int(num_blocks)
        self.observed_blocks = int(observed_blocks)
        self.sigma_y = float(sigma_y)
        self.observation_tilt = float(observation_tilt)
        self.banana_tau = float(banana_tau)
        self.sine_tau = float(sine_tau)
        self.prior_dim = cfg.dim
        self.active_dim = 2 * self.observed_blocks
        self.observation_values = observation_values[: self.observed_blocks].copy()

    @property
    def prior_signature(self):
        """Distribution identity used to reject incompatible checkpoints."""
        return {"geometry": GEOMETRY_VERSION, "scenario": self.scenario,
                "dim": self.cfg.dim, "banana_tau": self.banana_tau,
                "sine_tau": self.sine_tau}

    def full_coordinates(self, z: np.ndarray):
        """Validate full coordinates without applying any rotation or projection."""
        z = np.asarray(z, dtype=float)
        if z.ndim < 1 or z.shape[-1] != self.prior_dim:
            raise ValueError(f"expected final dimension {self.prior_dim}, got {z.shape}")
        return z

    def active(self, z: np.ndarray):
        """Consecutive source coordinates of the observed blocks."""
        return self.full_coordinates(z)[..., :self.active_dim]

    def transport_blocks(self, uv: np.ndarray):
        uv = np.asarray(uv, dtype=float)
        u, v = uv[..., 0], uv[..., 1]
        x = np.empty_like(uv)
        x[..., 0] = u
        if self.scenario == "banana":
            # Thin banana: conditionally on u, x2 has standard deviation tau.
            x[..., 1] = self.banana_tau * v + 0.62 * (u**2 - 1.0)
        else:
            # Thin sine ridge with the same configurable transverse scale.
            x[..., 1] = self.sine_tau * v + 1.05 * np.sin(1.65 * u)
        return x

    def transport(self, z: np.ndarray):
        """Transform every consecutive (u, v) pair directly in full coordinates."""
        z = self.full_coordinates(z)
        blocks = z.reshape(z.shape[:-1] + (self.num_blocks, 2))
        return self.transport_blocks(blocks).reshape(z.shape)

    def data_active(self, x: np.ndarray):
        """Consecutive data coordinates of the observed blocks."""
        return self.active(x)

    def observe_blocks(self, blocks: np.ndarray):
        """Apply the shared linear observation to data-space blocks.

        For a data block ``(x1, x2)``, the observation is
        ``x2 + observation_tilt * x1``.  Setting the tilt to zero exactly
        recovers the original experiment.
        """
        blocks = np.asarray(blocks)
        return blocks[..., 1] + self.observation_tilt * blocks[..., 0]

    def data_phi(self, x: np.ndarray):
        """Negative log-likelihood evaluated directly in data coordinates."""
        active = self.data_active(x)
        blocks = active.reshape(active.shape[:-1] + (self.observed_blocks, 2))
        residual = (
            self.observe_blocks(blocks) - self.observation_values
        ) / self.sigma_y
        return 0.5 * np.sum(residual**2, axis=-1)

    def transport_torch(self, z: "torch.Tensor"):
        """Differentiable torch implementation of :meth:`transport`."""
        active = self.full_coordinates_torch(z)
        blocks = active.reshape(active.shape[:-1] + (self.num_blocks, 2))
        u, v = blocks[..., 0], blocks[..., 1]
        transformed = torch.empty_like(blocks)
        transformed[..., 0] = u
        if self.scenario == "banana":
            transformed[..., 1] = (
                self.banana_tau * v + 0.62 * (u.square() - 1.0)
            )
        else:
            transformed[..., 1] = (
                self.sine_tau * v + 1.05 * torch.sin(1.65 * u)
            )
        transformed = transformed.reshape(active.shape)
        return transformed

    def full_coordinates_torch(self, x: "torch.Tensor"):
        if torch is None:
            raise ImportError("PyTorch is required for learned-transport experiments")
        if x.ndim < 1 or x.shape[-1] != self.prior_dim:
            raise ValueError(f"expected final dimension {self.prior_dim}, got {tuple(x.shape)}")
        return x

    def data_active_torch(self, x: "torch.Tensor"):
        return self.full_coordinates_torch(x)[..., :self.active_dim]

    def observe_blocks_torch(self, blocks: "torch.Tensor"):
        """Torch counterpart of :meth:`observe_blocks`."""
        return blocks[..., 1] + self.observation_tilt * blocks[..., 0]

    def data_phi_torch(self, x: "torch.Tensor"):
        active = self.data_active_torch(x)
        blocks = active.reshape(active.shape[:-1] + (self.observed_blocks, 2))
        observation = torch.as_tensor(
            self.observation_values, device=x.device, dtype=x.dtype
        )
        residual = (
            self.observe_blocks_torch(blocks) - observation
        ) / self.sigma_y
        return 0.5 * residual.square().sum(dim=-1)

    def sample_data_torch(
        self,
        samples: int,
        *,
        device: Any,
        dtype: "torch.dtype",
        generator: "torch.Generator | None" = None,
    ):
        """Draw sample-only prior data without exposing densities to learners."""
        if torch is None:
            raise ImportError("PyTorch is required for learned-transport experiments")
        source = torch.randn(
            int(samples), self.cfg.dim, device=device, dtype=dtype,
            generator=generator,
        )
        return self.transport_torch(source)

    def pair_phi_grad(
        self,
        uv: np.ndarray,
        observation_value: float | np.ndarray | None = None,
    ):
        uv = np.asarray(uv, dtype=float)
        u = uv[..., 0]
        x = self.transport_blocks(uv)
        if observation_value is None:
            observation_value = self.observation_values
        prediction = self.observe_blocks(x)
        residual = (prediction - np.asarray(observation_value)) / self.sigma_y
        value = 0.5 * residual**2
        dphi_dprediction = residual / self.sigma_y
        if self.scenario == "banana":
            dprediction_du = 1.24 * u + self.observation_tilt
            dprediction_dv = self.banana_tau
        else:
            dprediction_du = (
                1.05 * 1.65 * np.cos(1.65 * u) + self.observation_tilt
            )
            dprediction_dv = self.sine_tau
        gradient = np.stack(
            (
                dphi_dprediction * dprediction_du,
                dphi_dprediction * dprediction_dv,
            ),
            axis=-1,
        )
        return value, gradient

    def phi_active_grad(self, active: np.ndarray):
        shape = active.shape
        blocks = active.reshape(shape[:-1] + (self.observed_blocks, 2))
        values, gradients = self.pair_phi_grad(blocks)
        return np.sum(values, axis=-1), gradients.reshape(shape)

    def phi(self, z: np.ndarray):
        value, _ = self.phi_active_grad(self.active(z))
        return value

    def grad_phi(self, z: np.ndarray):
        _, gradient = self.phi_active_grad(self.active(z))
        full_gradient = np.zeros_like(self.full_coordinates(z))
        full_gradient[..., :self.active_dim] = gradient
        return full_gradient

    def prior_gn_lambda95(self, samples: int = 50_000, seed: int = 1801):
        """95th percentile of the single-block Gauss--Newton eigenvalue."""
        rng = np.random.default_rng(seed)
        u = rng.normal(size=samples)
        if self.scenario == "banana":
            derivative = 1.24 * u + self.observation_tilt
            vertical_derivative = self.banana_tau
        else:
            derivative = (
                1.05 * 1.65 * np.cos(1.65 * u) + self.observation_tilt
            )
            vertical_derivative = self.sine_tau
        eigenvalue = (
            vertical_derivative**2 + derivative**2
        ) / self.sigma_y**2
        return float(np.quantile(eigenvalue, 0.95))


def make_pair_reference(
    problem: CurvedObservationProblem,
    observation_value: float,
    grid_n: int,
    samples: int,
    seed: int,
):
    if grid_n < 2 or samples < 1:
        raise ValueError("reference needs grid_n >= 2 and samples >= 1")
    grid = np.linspace(-4.6, 4.6, grid_n)
    u, v = np.meshgrid(grid, grid, indexing="ij")
    uv = np.stack((u, v), axis=-1)
    phi, _ = problem.pair_phi_grad(uv, observation_value)
    log_weight = -0.5 * (u**2 + v**2) - phi
    log_weight -= np.max(log_weight)
    weight = np.exp(log_weight).reshape(-1)
    weight /= np.sum(weight)
    rng = np.random.default_rng(seed)
    indices = rng.choice(weight.size, size=samples, replace=True, p=weight)
    step = grid[1] - grid[0]
    pair = uv.reshape(-1, 2)[indices]
    pair += rng.uniform(-0.5 * step, 0.5 * step, size=pair.shape)
    data = problem.transport_blocks(pair)
    edges = np.quantile(pair[:, 0], np.linspace(0.0, 1.0, 9))
    edges[0], edges[-1] = -np.inf, np.inf
    labels = np.clip(np.digitize(pair[:, 0], edges[1:-1]), 0, 7)
    bin_weight = np.bincount(labels, minlength=8) / labels.size
    return pair, data, edges, bin_weight


def make_active_reference(
    pairs: list[np.ndarray],
    samples: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    blocks = []
    for pair in pairs:
        indices = rng.integers(0, len(pair), size=samples)
        blocks.append(pair[indices])
    return np.stack(blocks, axis=1).reshape(samples, 2 * len(pairs))


class LearnedTransportPosterior:
    """Torch posterior wrapper for a differentiable learned source-to-data map."""

    def __init__(self, base: CurvedObservationProblem, transport: Any):
        if torch is None:
            raise ImportError("PyTorch is required for learned-transport experiments")
        self.base = base
        self.transport = transport
        self.scenario = base.scenario
        self.device = transport.device
        self.dtype = transport.dtype

    def phi_torch(self, z: "torch.Tensor"):
        return self.base.data_phi_torch(self.transport(z))

    def phi_and_grad(
        self, z: "torch.Tensor"
    ):
        with torch.enable_grad():
            position = z.detach().requires_grad_(True)
            value = self.phi_torch(position)
            gradient = torch.autograd.grad(value.sum(), position)[0]
        return value.detach(), gradient.detach()

    def source_active_torch(self, z: "torch.Tensor"):
        return self.base.data_active_torch(z)

    def data_active_torch(self, z: "torch.Tensor"):
        return self.base.data_active_torch(self.transport(z))


def forward_observation(
    scenario: str,
    uv: np.ndarray,
    observation_tilt: float = 0.0,
    banana_tau: float = DEFAULT_BANANA_TAU,
    sine_tau: float = DEFAULT_SINE_TAU,
):
    u, v = uv[..., 0], uv[..., 1]
    if scenario == "banana":
        if not np.isfinite(banana_tau) or banana_tau <= 0:
            raise ValueError("banana_tau must be positive and finite")
        x2 = float(banana_tau) * v + 0.62 * (u**2 - 1.0)
    else:
        if not np.isfinite(sine_tau) or sine_tau <= 0:
            raise ValueError("sine_tau must be positive and finite")
        x2 = float(sine_tau) * v + 1.05 * np.sin(1.65 * u)
    return x2 + float(observation_tilt) * u


def make_datasets(
    scenario: str,
    num_blocks: int,
    mode: str,
    data_seeds: tuple[int, ...],
    fixed_observations: tuple[float, ...],
    generated_noise_sigma: float,
    observation_tilt: float = 0.0,
    banana_tau: float = DEFAULT_BANANA_TAU,
    sine_tau: float = DEFAULT_SINE_TAU,
):
    datasets: list[tuple[str, np.ndarray]] = []
    if mode == "fixed":
        for index, value in enumerate(fixed_observations):
            datasets.append((f"fixed_{index}_y{value:g}", np.full(num_blocks, value)))
        return datasets
    for seed in data_seeds:
        rng = np.random.default_rng(seed)
        truth = rng.normal(size=(num_blocks, 2))
        observation = forward_observation(
            scenario,
            truth,
            observation_tilt=observation_tilt,
            banana_tau=banana_tau,
            sine_tau=sine_tau,
        )
        if generated_noise_sigma > 0:
            observation = observation + generated_noise_sigma * rng.normal(size=num_blocks)
        datasets.append((f"prior_predictive_seed{seed}", observation))
    return datasets
