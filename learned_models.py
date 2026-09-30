"""Sample-only priors for the end-to-end source-sampling study.

iMF follows the supplied imf_auxhead_check.py recipe: raw (r,t) conditions,
three hidden layers by default in the driver, separate u/v heads, independent
uniform times, mean-squared losses, Adam and no gradient clipping. The v head
is evaluated at r=t and trained directly; both JVP tangent and correction are
detached. Four-step inference evaluates only the u head.

FM and diffusion retain their existing architectures, objectives and samplers.
Training exposes only prior samples, never the analytic reference transport.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, jvp
import torch.nn.functional as F


MODEL_IMPLEMENTATION_VERSION = {
    "imf": 7,       # supplied auxiliary-head recipe; incompatible with v6 weights
    "fm": 2,
    "diffusion": 2,
}


@dataclass(frozen=True)
class NetworkConfig:
    dim: int
    hidden: int = 512
    depth: int = 3
    time_harmonics: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 16_000
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    log_every: int = 500
    fm_coupling: str = "minibatch-ot"
    imf_coupling: str = "independent"
    imf_auxiliary_weight: float = 1.0
    # Defaults reproduce the supplied auxiliary-head experiment.
    imf_learning_rate: float = 1e-3
    imf_adam_beta1: float = 0.9
    imf_adam_beta2: float = 0.999
    imf_weight_decay: float = 1e-5
    imf_grad_clip: float = 0.0  # independent of baseline grad_clip
    imf_time_distribution: str = "uniform"
    imf_time_mean: float = -0.4  # only for the optional logit-normal ablation
    imf_time_std: float = 1.0
    imf_equal_time_fraction: float = 0.0
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _time_features(value: torch.Tensor, harmonics: int) -> torch.Tensor:
    value = value.reshape(-1, 1)
    if harmonics <= 0:
        return value
    frequencies = (2.0 ** torch.arange(
        harmonics, device=value.device, dtype=value.dtype
    )).reshape(1, -1) * math.pi
    angles = value * frequencies
    return torch.cat((value, torch.sin(angles), torch.cos(angles)), dim=-1)


class ConditionalMLP(nn.Module):
    """Plain feed-forward network with time features concatenated to the input.

    ``cfg.depth`` counts hidden layers, each Linear + SiLU with ``cfg.hidden``
    units. A final linear layer produces the requested output heads. There are
    no normalization layers or residual connections. SiLU remains smooth for
    iMF's JVP and the derivatives used by the downstream samplers.
    """

    def __init__(
        self,
        cfg: NetworkConfig,
        scalar_conditions: int,
        output_heads: int,
    ):
        super().__init__()
        if cfg.dim < 1 or cfg.hidden < 1 or cfg.depth < 1:
            raise ValueError("dim, hidden and depth must be positive")
        if cfg.time_harmonics < 0 or scalar_conditions < 1 or output_heads < 1:
            raise ValueError("invalid time harmonics, scalar conditions or output heads")
        self.cfg = cfg
        self.scalar_conditions = int(scalar_conditions)
        self.output_heads = int(output_heads)
        condition_dim = self.scalar_conditions * (1 + 2 * cfg.time_harmonics)
        layers = [nn.Linear(cfg.dim + condition_dim, cfg.hidden), nn.SiLU()]
        for _ in range(cfg.depth - 1):
            layers.extend((nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU()))
        self.hidden_layers = nn.Sequential(*layers)
        self.output = nn.Linear(cfg.hidden, cfg.dim * self.output_heads)

    def forward(
        self, x: torch.Tensor, *conditions: torch.Tensor
    ):
        if len(conditions) != self.scalar_conditions:
            raise ValueError(
                f"expected {self.scalar_conditions} scalar conditions, "
                f"received {len(conditions)}"
            )
        features = [x]
        features.extend(
            _time_features(condition, self.cfg.time_harmonics)
            for condition in conditions
        )
        hidden = self.hidden_layers(torch.cat(features, dim=-1))
        output = self.output(hidden)
        if self.output_heads == 1:
            return output
        return tuple(output.chunk(self.output_heads, dim=-1))


# Preserve imports of the previous class name; this alias uses the new MLP.
ConditionalResidualMLP = ConditionalMLP


class IMFAuxiliaryMLP(nn.Module):
    """Supplied iMFNet architecture; forward(z,r,t) returns only average velocity.

    cfg.depth counts hidden Linear+SiLU layers. Time inputs are raw scalars;
    cfg.time_harmonics is unused by this family (driver records it as zero).
    Separate head_v is supervised at (z,t,t), and never used during sampling.
    """
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        if min(cfg.dim, cfg.hidden, cfg.depth) < 1:
            raise ValueError("dim, hidden and depth must be positive")
        self.cfg = cfg
        layers = [nn.Linear(cfg.dim + 2, cfg.hidden), nn.SiLU()]
        for _ in range(cfg.depth - 1):
            layers.extend((nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU()))
        self.backbone = nn.Sequential(*layers)
        self.head_u = nn.Linear(cfg.hidden, cfg.dim)
        self.head_v = nn.Linear(cfg.hidden, cfg.dim)

    def forward(self, z, r, t):
        h = self.backbone(torch.cat((z, r.reshape(-1, 1), t.reshape(-1, 1)), dim=-1))
        return self.head_u(h)

    def get_v(self, z, t):
        t = t.reshape(-1, 1)
        h = self.backbone(torch.cat((z, t, t), dim=-1))
        return self.head_v(h)


def build_model(kind: str, cfg: NetworkConfig):
    if kind == "imf":
        return IMFAuxiliaryMLP(cfg)
    if kind in {"fm", "diffusion"}:
        return ConditionalMLP(cfg, scalar_conditions=1, output_heads=1)
    raise ValueError(f"unknown learned-prior kind {kind!r}")


def _sample_imf_times(
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unsorted independent U[0,1] times by default; alternatives are ablations."""
    if not 0.0 <= cfg.imf_equal_time_fraction <= 1.0:
        raise ValueError("imf_equal_time_fraction must lie in [0,1]")
    law = cfg.imf_time_distribution
    if law in {"uniform", "sorted-uniform"}:
        # Match the supplied script's draw order and tensor shapes exactly.
        t = torch.rand(batch, 1, device=device, dtype=dtype).reshape(-1)
        r = torch.rand(batch, 1, device=device, dtype=dtype).reshape(-1)
        if law == "sorted-uniform":
            t, r = torch.maximum(t, r), torch.minimum(t, r)
    elif law == "logit-normal":
        if cfg.imf_time_std <= 0:
            raise ValueError("imf_time_std must be positive")
        logits = torch.randn(batch, 2, device=device, dtype=dtype)
        times = torch.sigmoid(logits * cfg.imf_time_std + cfg.imf_time_mean)
        t, r = times.max(dim=1).values, times.min(dim=1).values
    else:
        raise ValueError(f"Unknown iMF time distribution: {law}")
    equal_count = int(batch * cfg.imf_equal_time_fraction)
    if equal_count:
        r = r.clone()
        r[:equal_count] = t[:equal_count]
    return t, r


def _minibatch_ot_pair(noise: torch.Tensor, data: torch.Tensor):
    """Return a hard minibatch-OT permutation using the Hungarian solver."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise ImportError(
            "scipy is required for minibatch-ot coupling"
        ) from error
    with torch.no_grad():
        cost = torch.cdist(noise.float(), data.float()).square().cpu().numpy()
        row, column = linear_sum_assignment(cost)
        order = np.empty(len(row), dtype=np.int64)
        order[row] = column
    return data[torch.as_tensor(order, device=data.device)]


def _imf_loss(
    model: nn.Module,
    data: torch.Tensor,
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if cfg.imf_auxiliary_weight <= 0.0:
        raise ValueError("imf_auxiliary_weight must be positive")
    t, r = _sample_imf_times(len(data), data.device, data.dtype, cfg)
    t, r = t[:, None], r[:, None]
    noise = torch.randn_like(data)
    if cfg.imf_coupling == "minibatch-ot":
        data = _minibatch_ot_pair(noise, data)
    elif cfg.imf_coupling != "independent":
        raise ValueError("imf_coupling must be independent or minibatch-ot")
    state = (1 - t) * data + t * noise
    target = noise - data
    velocity = model.get_v(state, t)
    params = dict(model.named_parameters())

    def average_velocity(z_in, r_in, t_in):
        return functional_call(model, params, (z_in, r_in, t_in))

    average, derivative = jvp(
        average_velocity, (state, r, t),
        (velocity.detach(), torch.zeros_like(r), torch.ones_like(t)),
    )
    reconstructed = average + (t - r) * derivative.detach()
    meanflow_mse = F.mse_loss(reconstructed, target)
    auxiliary_mse = F.mse_loss(velocity, target)
    loss = meanflow_mse + cfg.imf_auxiliary_weight * auxiliary_mse
    return loss, {
        "meanflow_mse": float(meanflow_mse.detach()),
        "auxiliary_velocity_mse": float(auxiliary_mse.detach()),
        # Sum-reduction diagnostics retained for theoretical/report compatibility.
        "meanflow_l2": float(meanflow_mse.detach()) * data.shape[-1],
        "auxiliary_velocity_l2": float(auxiliary_mse.detach()) * data.shape[-1],
    }


def _fm_loss(
    model: nn.Module,
    data: torch.Tensor,
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    noise = torch.randn_like(data)
    if cfg.fm_coupling == "minibatch-ot":
        data = _minibatch_ot_pair(noise, data)
    elif cfg.fm_coupling != "independent":
        raise ValueError("fm_coupling must be independent or minibatch-ot")
    t = torch.rand(data.shape[0], device=data.device, dtype=data.dtype)
    state = (1.0 - t[:, None]) * data + t[:, None] * noise
    target = noise - data
    loss = F.mse_loss(model(state, t), target)
    return loss, {"flow_loss": float(loss.detach())}


def vp_schedule(
    steps: int,
    beta_start: float,
    beta_end: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(steps) < 2:
        raise ValueError("diffusion_steps must be at least 2")
    betas = torch.linspace(beta_start, beta_end, int(steps), device=device, dtype=dtype)
    betas = betas.clamp(1e-8, 0.999)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


def _diffusion_loss(
    model: nn.Module,
    data: torch.Tensor,
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    _, _, alpha_bars = vp_schedule(
        cfg.diffusion_steps, cfg.beta_start, cfg.beta_end,
        device=data.device, dtype=data.dtype,
    )
    index = torch.randint(
        0, cfg.diffusion_steps, (data.shape[0],), device=data.device
    )
    noise = torch.randn_like(data)
    alpha_bar = alpha_bars[index, None]
    state = alpha_bar.sqrt() * data + (1.0 - alpha_bar).sqrt() * noise
    time_condition = (index.to(data.dtype) + 1.0) / cfg.diffusion_steps
    loss = F.mse_loss(model(state, time_condition), noise)
    return loss, {"epsilon_loss": float(loss.detach())}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_learned_prior(
    kind: str,
    model: nn.Module,
    data: torch.Tensor,
    cfg: TrainingConfig,
    *,
    seed: int,
) -> dict[str, object]:
    """Train one prior model with a fixed sample dataset and update budget."""
    if cfg.steps < 1 or cfg.batch_size < 1 or len(data) < 1:
        raise ValueError("steps, batch size, and dataset size must be positive")
    torch.manual_seed(int(seed))
    if data.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    if kind == "imf":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.imf_learning_rate,
            betas=(cfg.imf_adam_beta1, cfg.imf_adam_beta2),
            weight_decay=cfg.imf_weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    loss_function: Callable = {
        "imf": _imf_loss,
        "fm": _fm_loss,
        "diffusion": _diffusion_loss,
    }[kind]
    model.train()
    losses: list[float] = []
    component_history: dict[str, list[float]] = {}
    components: dict[str, float] = {}
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_003)
    _synchronize(data.device)
    start = time.perf_counter()
    clipping_threshold = cfg.imf_grad_clip if kind == "imf" else cfg.grad_clip
    batch_start = len(data)
    permutation = None
    examples_seen = 0
    for step in range(cfg.steps):
        if kind == "imf":
            # Shuffle each epoch and include the final short batch, as supplied.
            if batch_start >= len(data):
                permutation = torch.randperm(len(data), generator=generator)
                batch_start = 0
            indices = permutation[batch_start:batch_start + cfg.batch_size].to(data.device)
            batch_start += cfg.batch_size
        else:
            indices = torch.randint(
                0, len(data), (cfg.batch_size,), generator=generator,
            ).to(data.device)
        batch = data[indices]
        loss, components = loss_function(model, batch, cfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clipping_threshold > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clipping_threshold, error_if_nonfinite=True,
            )
        else:
            gradient_norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad.detach())
                for p in model.parameters() if p.grad is not None
            ]))
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"Nonfinite {kind} loss/gradient at step {step + 1}")
        grad_value = float(gradient_norm)
        clip_scale = min(1.0, clipping_threshold / (grad_value + 1e-6)) if clipping_threshold > 0 else 1.0
        components.update(grad_norm=grad_value, clip_scale=clip_scale,
                          clipped=float(clip_scale < 1.0))
        examples_seen += len(batch)
        optimizer.step()
        losses.append(float(loss.detach()))
        for key, value in components.items():
            component_history.setdefault(key, []).append(float(value))
        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            details = " ".join(f"{key}={value:.4g}" for key, value in components.items())
            print(
                f"{kind:9s} step {step + 1:6d}/{cfg.steps} "
                f"loss={losses[-1]:.4g} {details}",
                flush=True,
            )
    _synchronize(data.device)
    elapsed = time.perf_counter() - start
    model.eval()
    metadata: dict[str, object] = {
        "kind": kind,
        "implementation_version": MODEL_IMPLEMENTATION_VERSION[kind],
        "training_time_sec": elapsed,
        "parameter_count": parameter_count(model),
        "final_loss": losses[-1],
        "mean_last_100_loss": float(np.mean(losses[-100:])),
        "gradient_norm_median": float(np.median(component_history["grad_norm"])),
        "gradient_norm_p99": float(np.quantile(component_history["grad_norm"], 0.99)),
        "fraction_clipped": float(np.mean(component_history["clipped"])),
        "mean_clip_scale": float(np.mean(component_history["clip_scale"])),
        "gradient_clip_threshold": clipping_threshold,
        "examples_seen": examples_seen,
        "effective_epochs": examples_seen / len(data),
        "batch_sampling": "shuffled epochs including final short batch" if kind == "imf" else "with replacement",
        "optimizer": "Adam" if kind == "imf" else "AdamW",
        "network_config": asdict(model.cfg),
        "training_config": asdict(cfg),
    }
    for key, values in component_history.items():
        metadata[f"final_{key}"] = values[-1]
        metadata[f"mean_last_100_{key}"] = float(np.mean(values[-100:]))
    return metadata


class LearnedTransport:
    """Differentiable source-to-data map backed by iMF or flow matching."""

    def __init__(
        self,
        kind: str,
        model: nn.Module,
        *,
        steps: int,
        solver: str = "midpoint",
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        if kind not in {"imf", "fm"}:
            raise ValueError("LearnedTransport supports imf or fm")
        if steps < 1:
            raise ValueError("transport steps must be positive")
        if solver not in {"euler", "midpoint", "rk4", "dopri5"}:
            raise ValueError("solver must be euler, midpoint, rk4, or dopri5")
        self.kind = kind
        self.model = model
        self.steps = int(steps)
        self.solver = solver
        self.rtol, self.atol = float(rtol), float(atol)
        self.function_evaluations = 0
        self.map_calls = 0
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype

    @property
    def nfe_per_call(self) -> float:
        if self.kind == "imf":
            return float(self.steps)
        if self.solver == "dopri5":
            return self.function_evaluations / max(self.map_calls, 1)
        return self.steps * {"euler": 1, "midpoint": 2, "rk4": 4}[self.solver]

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if source.ndim < 2:
            raise ValueError("transport input must have shape (..., dim)")
        if source.shape[-1] != self.model.cfg.dim:
            raise ValueError(
                f"transport expected final dimension {self.model.cfg.dim}, "
                f"received {source.shape[-1]}"
            )
        # Samplers naturally carry several batch axes, e.g.
        # (replicas, particles, dim) or (sweeps, chains, dim).  The MLP uses a
        # conventional two-dimensional batch, so collapse every leading axis
        # for integration and restore it before returning.
        original_shape = source.shape
        state = source.reshape(-1, source.shape[-1])
        self.map_calls += 1
        if self.kind == "imf":
            grid = torch.linspace(
                1.0, 0.0, self.steps + 1,
                device=state.device, dtype=state.dtype,
            )
            for index in range(self.steps):
                t_value, r_value = grid[index], grid[index + 1]
                t = torch.full((state.shape[0],), t_value, device=state.device, dtype=state.dtype)
                r = torch.full_like(t, r_value)
                average = self.model(state, r, t)
                self.function_evaluations += 1
                state = state - (t_value - r_value) * average
            return state.reshape(original_shape)


        if self.solver == "dopri5":
            try:
                from torchdiffeq import odeint
            except ImportError as error:
                raise ImportError(
                    "--fm-solver dopri5 requires `pip install torchdiffeq`"
                ) from error

            def vector_field(t_scalar: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
                self.function_evaluations += 1
                condition = t_scalar.to(
                    device=value.device, dtype=value.dtype
                ).expand(value.shape[0])
                return self.model(value, condition)

            output_times = torch.linspace(
                1.0, 0.0, self.steps + 1,
                device=state.device, dtype=state.dtype,
            )
            state = odeint(
                vector_field,
                state,
                output_times,
                rtol=self.rtol,
                atol=self.atol,
                method="dopri5",
            )[-1]
            return state.reshape(original_shape)

        step = -1.0 / self.steps
        for index in range(self.steps):
            t_value = 1.0 - index / self.steps
            t = torch.full((state.shape[0],), t_value, device=state.device, dtype=state.dtype)
            if self.solver == "euler":
                self.function_evaluations += 1
                state = state + step * self.model(state, t)
            elif self.solver == "midpoint":
                self.function_evaluations += 2
                k1 = self.model(state, t)
                middle_t = torch.full_like(t, t_value + 0.5 * step)
                k2 = self.model(state + 0.5 * step * k1, middle_t)
                state = state + step * k2
            else:
                self.function_evaluations += 4
                k1 = self.model(state, t)
                middle_t = torch.full_like(t, t_value + 0.5 * step)
                k2 = self.model(state + 0.5 * step * k1, middle_t)
                k3 = self.model(state + 0.5 * step * k2, middle_t)
                end_t = torch.full_like(t, t_value + step)
                k4 = self.model(state + step * k3, end_t)
                state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return state.reshape(original_shape)


@torch.no_grad()
def sample_learned_prior(
    transport: LearnedTransport,
    samples: int,
    dim: int,
    *,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=transport.device).manual_seed(int(seed))
    output = []
    for start in range(0, int(samples), int(batch_size)):
        batch = min(int(batch_size), int(samples) - start)
        source = torch.randn(
            batch, dim, generator=generator,
            device=transport.device, dtype=transport.dtype,
        )
        output.append(transport(source).cpu())
    return torch.cat(output, dim=0)


def save_checkpoint(
    path: Path,
    kind: str,
    model: nn.Module,
    network_cfg: NetworkConfig,
    training_cfg: TrainingConfig,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": kind,
        "implementation_version": MODEL_IMPLEMENTATION_VERSION[kind],
        "network_config": asdict(network_cfg),
        "training_config": asdict(training_cfg),
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }, path)


def load_checkpoint(path: Path, device: torch.device, dtype: torch.dtype):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    kind = checkpoint["kind"]
    saved_version = int(checkpoint.get("implementation_version", 1))
    if saved_version != MODEL_IMPLEMENTATION_VERSION[kind]:
        raise ValueError(
            f"Checkpoint {path} uses {kind} implementation version {saved_version}; "
            f"the current implementation requires version {MODEL_IMPLEMENTATION_VERSION[kind]}. "
            "Retrain with --no-reuse-checkpoints or use a new checkpoint directory."
        )
    network_cfg = NetworkConfig(**checkpoint["network_config"])
    # Ignore retired configuration fields only after checking architecture compatibility.
    valid_training_fields = {field.name for field in fields(TrainingConfig)}
    training_cfg = TrainingConfig(**{
        key: value
        for key, value in checkpoint["training_config"].items()
        if key in valid_training_fields
    })
    model = build_model(checkpoint["kind"], network_cfg).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = dict(checkpoint.get("metadata", {}))
    metadata["implementation_version"] = int(
        checkpoint.get("implementation_version", 1)
    )
    return checkpoint["kind"], model, network_cfg, training_cfg, metadata


def _posterior_variance(
    beta: torch.Tensor, alpha_bar: torch.Tensor, alpha_bar_previous: torch.Tensor
) -> torch.Tensor:
    return beta * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar).clamp_min(1e-12)


@torch.no_grad()
def sample_diffusion_prior(
    model: nn.Module,
    *,
    samples: int,
    dim: int,
    training_cfg: TrainingConfig,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    """Unconditional DDPM ancestral sampling for prior-fit diagnostics."""
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    betas, alphas, alpha_bars = vp_schedule(
        training_cfg.diffusion_steps,
        training_cfg.beta_start,
        training_cfg.beta_end,
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    output = []
    for start in range(0, int(samples), int(batch_size)):
        batch = min(int(batch_size), int(samples) - start)
        state = torch.randn(
            batch, dim, generator=generator, device=device, dtype=dtype
        )
        for index in range(training_cfg.diffusion_steps - 1, -1, -1):
            time_condition = torch.full(
                (batch,), (index + 1.0) / training_cfg.diffusion_steps,
                device=device, dtype=dtype,
            )
            epsilon = model(state, time_condition)
            beta, alpha, alpha_bar = (
                betas[index], alphas[index], alpha_bars[index]
            )
            root_one_minus = (1.0 - alpha_bar).sqrt().clamp_min(1e-12)
            mean = (
                state - beta / root_one_minus * epsilon
            ) / alpha.sqrt().clamp_min(1e-12)
            if index > 0:
                variance = _posterior_variance(
                    beta, alpha_bar, alpha_bars[index - 1]
                ).clamp_min(1e-12)
                noise = torch.randn(
                    state.shape, generator=generator,
                    device=device, dtype=dtype,
                )
                state = mean + variance.sqrt() * noise
            else:
                state = (
                    state - root_one_minus * epsilon
                ) / alpha_bar.sqrt().clamp_min(1e-12)
        output.append(state.cpu())
    return torch.cat(output, dim=0)


def dps_sample(
    model: nn.Module,
    problem,
    *,
    samples: int,
    dim: int,
    training_cfg: TrainingConfig,
    guidance_scale: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate DPS samples using the original residual-norm guidance.

    For Gaussian measurements, the DPS2022 implementation differentiates the
    norm of the measurement residual through the Tweedie estimate and subtracts
    ``scale * grad(norm)``.  It does not use the gradient of the Gaussian
    log-likelihood.  We compute the norm independently for every batch item so
    the result is invariant to the diagnostic batch size.
    """
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    betas, alphas, alpha_bars = vp_schedule(
        training_cfg.diffusion_steps,
        training_cfg.beta_start,
        training_cfg.beta_end,
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    outputs: list[torch.Tensor] = []
    _synchronize(device)
    start_time = time.perf_counter()
    for start in range(0, int(samples), int(batch_size)):
        batch = min(int(batch_size), int(samples) - start)
        state = torch.randn(
            batch, dim, generator=generator, device=device, dtype=dtype
        )
        for index in range(training_cfg.diffusion_steps - 1, -1, -1):
            with torch.enable_grad():
                position = state.detach().requires_grad_(True)
                t = torch.full(
                    (batch,), (index + 1.0) / training_cfg.diffusion_steps,
                    device=device, dtype=dtype,
                )
                epsilon = model(position, t)
                alpha_bar = alpha_bars[index]
                root_one_minus = (1.0 - alpha_bar).sqrt().clamp_min(1e-12)
                x0 = (position - root_one_minus * epsilon) / alpha_bar.sqrt().clamp_min(1e-12)
                active = problem.data_active_torch(x0).reshape(
                    batch, problem.observed_blocks, 2
                )
                observation = torch.as_tensor(
                    problem.observation_values,
                    device=device,
                    dtype=dtype,
                )
                prediction = problem.observe_blocks_torch(active)
                residual = prediction - observation
                residual_norm = torch.linalg.vector_norm(residual, dim=1)
                guidance = torch.autograd.grad(
                    residual_norm.sum(), position
                )[0]
            beta, alpha = betas[index], alphas[index]
            mean = (
                position.detach()
                - beta / root_one_minus * epsilon.detach()
            ) / alpha.sqrt().clamp_min(1e-12)
            if index > 0:
                variance = _posterior_variance(
                    beta, alpha_bar, alpha_bars[index - 1]
                ).clamp_min(1e-12)
                noise = torch.randn(
                    state.shape, generator=generator, device=device, dtype=dtype
                )
                state = mean + variance.sqrt() * noise
            else:
                state = x0.detach()
            state = state - float(guidance_scale) * guidance.detach()
        outputs.append(state.detach().cpu())
    _synchronize(device)
    return torch.cat(outputs, dim=0).numpy(), {
        "runtime_sec": time.perf_counter() - start_time,
        "diffusion_steps": training_cfg.diffusion_steps,
        "guidance_scale": float(guidance_scale),
        "guidance_objective": "per-sample measurement-residual norm",
    }
