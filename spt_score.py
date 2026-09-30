"""SPT kernels, adaptation, and Gaussian-reference split HMC."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import torch
except ImportError:  # NumPy-only crossover experiments remain supported.
    torch = None

if TYPE_CHECKING:
    from curved_problem import CurvedObservationProblem


@dataclass(frozen=True)
class SamplerConfig:
    """Immutable settings consumed by the SPT samplers.

    The command-line interface owns experiment choices.  This object contains
    only the resolved settings for one sampler run.  Named constructors below
    make condition-specific changes explicit at the call site.
    """

    dim: int = 256
    replicas: int = 24
    particles: int = 10
    adapt_sweeps: int = 500
    burnin_sweeps: int = 500
    retained_sweeps: int = 600
    thin: int = 1
    beta_hmc: float = 1.0
    hmc_steps: int = 6
    cold_hmc_probability: float = 1.0

    def with_hmc_steps(self, hmc_steps: int) -> "SamplerConfig":
        """Return the configuration for a condition-specific HMC trajectory."""
        return replace(self, hmc_steps=int(hmc_steps))

    def with_retained_sweeps(self, retained_sweeps: int) -> "SamplerConfig":
        """Return an equal-cost run configuration with a new retained length."""
        return replace(self, retained_sweeps=int(retained_sweeps))

    def with_cold_hmc_probability(
        self, cold_hmc_probability: float
    ) -> "SamplerConfig":
        """Return a kernel-mixture variant without changing other settings."""
        return replace(self, cold_hmc_probability=float(cold_hmc_probability))


@dataclass
class RunResult:
    method: str
    scenario: str
    seed: int
    active_trace: np.ndarray
    local_acceptance: np.ndarray
    swap_acceptance: np.ndarray
    proposal_scale: np.ndarray
    runtime_sec: float
    phi_evaluations: int
    gradient_evaluations: int
    hmc_mask: np.ndarray
    hmc_acceptance: np.ndarray
    pcn_acceptance: np.ndarray
    hmc_move_fraction: np.ndarray
    data_active_trace: np.ndarray | None = None
    timing_sec: dict[str, float] | None = None
    retained_per_chain: int = 0
    retained_chains: int = 0


class SPT:
    """Standalone SPT core for pCN and a cold pCN/split-HMC mixture."""

    def __init__(self, problem, cfg: SamplerConfig, method: str, seed: int):
        if method not in {"pcn", "full_split_hmc_mixture"}:
            raise ValueError(f"unsupported method {method!r}")
        if not 0.0 <= cfg.cold_hmc_probability <= 1.0:
            raise ValueError("cold_hmc_probability must lie in [0, 1]")
        self.problem, self.cfg, self.method = problem, cfg, method
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        u = np.linspace(0.0, 1.0, cfg.replicas)
        self.betas = u**2
        self.betas[0], self.betas[-1] = 0.0, 1.0
        self.hmc_mask = (
            self.betas >= cfg.beta_hmc
            if method == "full_split_hmc_mixture"
            else np.zeros(cfg.replicas, dtype=bool)
        )
        self.eta = 0.92 * (1.0 - self.betas) + 0.06 * self.betas
        self.eta[0] = 0.999
        self.logit_eta = np.log(self.eta / (1.0 - self.eta))
        self.epsilon = 0.080 * (1.0 - self.betas) + 0.050 * self.betas
        self.log_epsilon = np.log(self.epsilon)
        self.phi_evaluations = 0
        self.gradient_evaluations = 0

    def _phi(self, z: np.ndarray) -> np.ndarray:
        self.phi_evaluations += int(np.prod(z.shape[:-1]))
        return self.problem.phi(z)

    def _phi_grad(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.gradient_evaluations += int(np.prod(z.shape[:-1]))
        return self.problem.phi(z), self.problem.grad_phi(z)

    def _adapt(
        self,
        pcn_acceptance: np.ndarray,
        hmc_acceptance: np.ndarray,
        sweep: int,
    ) -> None:
        """Default shared adapter; benchmark subclasses use the same rule."""
        rate = 0.10 / math.sqrt(1.0 + sweep / 30.0)
        target = np.full(self.cfg.replicas, 0.30)
        target[0] = 1.0
        pcn = np.isfinite(pcn_acceptance)
        self.logit_eta[pcn] += rate * (pcn_acceptance[pcn] - target[pcn])
        self.logit_eta = np.clip(self.logit_eta, -7.0, 6.9)
        self.eta = 1.0 / (1.0 + np.exp(-self.logit_eta))
        self.eta[0] = 0.999

        hrate = 0.055 / math.sqrt(1.0 + sweep / 30.0)
        hmc = np.isfinite(hmc_acceptance)
        self.log_epsilon[hmc] += hrate * (hmc_acceptance[hmc] - 0.80)
        self.log_epsilon = np.clip(
            self.log_epsilon, math.log(0.0015), math.log(0.22)
        )
        self.epsilon = np.exp(self.log_epsilon)

    def _pcn(
        self,
        z: np.ndarray,
        phi: np.ndarray,
        beta: float,
        eta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        proposal = (
            math.sqrt(max(0.0, 1.0 - eta**2)) * z
            + eta * self.rng.normal(size=z.shape)
        )
        phi_prop = self._phi(proposal)
        accept = np.log(self.rng.random(z.shape[0])) < np.minimum(
            0.0, -beta * (phi_prop - phi)
        )
        out, out_phi = z.copy(), phi.copy()
        out[accept], out_phi[accept] = proposal[accept], phi_prop[accept]
        return out, out_phi, accept

    def _full_hmc(
        self,
        z: np.ndarray,
        phi: np.ndarray,
        beta: float,
        epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MH-corrected Gaussian-reference split HMC in all coordinates."""
        z_prop = z.copy()
        momentum0 = self.rng.normal(size=z.shape)
        momentum = momentum0.copy()
        phi_prop, grad = self._phi_grad(z_prop)
        momentum -= 0.5 * epsilon * beta * grad
        cosine, sine = math.cos(epsilon), math.sin(epsilon)
        for leap in range(self.cfg.hmc_steps):
            old_z = z_prop
            z_prop = cosine * old_z + sine * momentum
            momentum = -sine * old_z + cosine * momentum
            phi_prop, grad = self._phi_grad(z_prop)
            kick = 0.5 if leap == self.cfg.hmc_steps - 1 else 1.0
            momentum -= kick * epsilon * beta * grad
        current_h = 0.5 * np.sum(z**2 + momentum0**2, axis=1) + beta * phi
        proposal_h = (
            0.5 * np.sum(z_prop**2 + momentum**2, axis=1) + beta * phi_prop
        )
        accept = np.log(self.rng.random(z.shape[0])) < np.minimum(
            0.0, current_h - proposal_h
        )
        out, out_phi = z.copy(), phi.copy()
        out[accept], out_phi[accept] = z_prop[accept], phi_prop[accept]
        return out, out_phi, accept

    def run(self) -> RunResult:
        c = self.cfg
        z = self.rng.normal(size=(c.replicas, c.particles, c.dim))
        phi = self._phi(z)
        total = c.adapt_sweeps + c.burnin_sweeps + c.retained_sweeps * c.thin
        local_accept = np.zeros(c.replicas)
        local_total = np.zeros(c.replicas)
        hmc_accept = np.zeros(c.replicas)
        hmc_total = np.zeros(c.replicas)
        pcn_accept = np.zeros(c.replicas)
        pcn_total = np.zeros(c.replicas)
        swap_accept = np.zeros(c.replicas - 1)
        swap_total = np.zeros(c.replicas - 1)
        traces: list[np.ndarray] = []
        start = time.perf_counter()

        for sweep in range(total):
            sweep_hmc_accept = np.full(c.replicas, np.nan)
            sweep_pcn_accept = np.full(c.replicas, np.nan)
            for k, beta in enumerate(self.betas):
                if self.hmc_mask[k]:
                    # A fixed-size random subset makes the realized mixture
                    # weight and the oracle-call budget deterministic.
                    n_hmc = int(round(c.particles * c.cold_hmc_probability))
                    use_hmc = np.zeros(c.particles, dtype=bool)
                    if n_hmc:
                        chosen = self.rng.choice(
                            c.particles, size=n_hmc, replace=False
                        )
                        use_hmc[chosen] = True
                    accepted = np.zeros(c.particles, dtype=bool)
                    if np.any(use_hmc):
                        z_hmc, phi_hmc, accepted_hmc = self._full_hmc(
                            z[k, use_hmc],
                            phi[k, use_hmc],
                            float(beta),
                            float(self.epsilon[k]),
                        )
                        z[k, use_hmc], phi[k, use_hmc] = z_hmc, phi_hmc
                        accepted[use_hmc] = accepted_hmc
                        sweep_hmc_accept[k] = float(np.mean(accepted_hmc))
                        if sweep >= c.adapt_sweeps:
                            hmc_accept[k] += np.sum(accepted_hmc)
                            hmc_total[k] += accepted_hmc.size
                    use_pcn = ~use_hmc
                    if np.any(use_pcn):
                        z_pcn, phi_pcn, accepted_pcn = self._pcn(
                            z[k, use_pcn],
                            phi[k, use_pcn],
                            float(beta),
                            float(self.eta[k]),
                        )
                        z[k, use_pcn], phi[k, use_pcn] = z_pcn, phi_pcn
                        accepted[use_pcn] = accepted_pcn
                        sweep_pcn_accept[k] = float(np.mean(accepted_pcn))
                        if sweep >= c.adapt_sweeps:
                            pcn_accept[k] += np.sum(accepted_pcn)
                            pcn_total[k] += accepted_pcn.size
                else:
                    z[k], phi[k], accepted = self._pcn(
                        z[k], phi[k], float(beta), float(self.eta[k])
                    )
                    sweep_pcn_accept[k] = float(np.mean(accepted))
                    if sweep >= c.adapt_sweeps:
                        pcn_accept[k] += np.sum(accepted)
                        pcn_total[k] += accepted.size

                if sweep >= c.adapt_sweeps:
                    local_accept[k] += np.sum(accepted)
                    local_total[k] += c.particles

            for parity in (sweep % 2, 1 - sweep % 2):
                for k in range(parity, c.replicas - 1, 2):
                    log_ratio = (self.betas[k + 1] - self.betas[k]) * (
                        phi[k + 1] - phi[k]
                    )
                    accepted = np.log(self.rng.random(c.particles)) < np.minimum(
                        0.0, log_ratio
                    )
                    if np.any(accepted):
                        temp = z[k, accepted].copy()
                        z[k, accepted] = z[k + 1, accepted]
                        z[k + 1, accepted] = temp
                        temp_phi = phi[k, accepted].copy()
                        phi[k, accepted] = phi[k + 1, accepted]
                        phi[k + 1, accepted] = temp_phi
                    if sweep >= c.adapt_sweeps:
                        swap_accept[k] += np.sum(accepted)
                        swap_total[k] += c.particles

            if sweep < c.adapt_sweeps:
                self._adapt(sweep_pcn_accept, sweep_hmc_accept, sweep)
                continue
            index = sweep - c.adapt_sweeps - c.burnin_sweeps
            if index >= 0 and index % c.thin == 0:
                # Observed-prefix access is a view; retain an independent snapshot.
                traces.append(self.problem.active(z[-1]).copy())

        scale = np.where(self.hmc_mask, self.epsilon, self.eta)
        return RunResult(
            method=self.method,
            scenario=self.problem.scenario,
            seed=self.seed,
            active_trace=np.stack(traces),
            local_acceptance=local_accept / np.maximum(local_total, 1.0),
            swap_acceptance=swap_accept / np.maximum(swap_total, 1.0),
            proposal_scale=scale,
            runtime_sec=time.perf_counter() - start,
            phi_evaluations=self.phi_evaluations,
            gradient_evaluations=self.gradient_evaluations,
            hmc_mask=self.hmc_mask.copy(),
            hmc_acceptance=np.divide(
                hmc_accept,
                hmc_total,
                out=np.full_like(hmc_accept, np.nan),
                where=hmc_total > 0,
            ),
            pcn_acceptance=np.divide(
                pcn_accept,
                pcn_total,
                out=np.full_like(pcn_accept, np.nan),
                where=pcn_total > 0,
            ),
            hmc_move_fraction=np.divide(
                hmc_total,
                hmc_total + pcn_total,
                out=np.zeros_like(hmc_total),
                where=(hmc_total + pcn_total) > 0,
            ),
        )


def adapt_pcn_shared(sampler: SPT, pcn_acceptance: np.ndarray, sweep: int) -> None:
    """One pCN adaptation implementation shared by every experimental arm."""
    rate = 0.10 / math.sqrt(1.0 + sweep / 30.0)
    target = np.full(sampler.cfg.replicas, 0.30)
    target[0] = 1.0
    finite = np.isfinite(pcn_acceptance)
    sampler.logit_eta[finite] += rate * (pcn_acceptance[finite] - target[finite])
    sampler.logit_eta = np.clip(sampler.logit_eta, -7.0, 6.9)
    sampler.eta = 1.0 / (1.0 + np.exp(-sampler.logit_eta))
    sampler.eta[0] = 0.999


class SharedAdapterPCNSPT(SPT):
    """Pure pCN arm using the exact pCN adapter used by the hybrid."""

    def _adapt(self, pcn_acceptance: np.ndarray, hmc_acceptance: np.ndarray, sweep: int) -> None:
        del hmc_acceptance
        adapt_pcn_shared(self, pcn_acceptance, sweep)


class TunedScheduleMixtureSPT(SPT):
    """Initialize from curvature, then adapt HMC steps during warmup only."""

    def __init__(
        self,
        *args,
        hmc_epsilon_schedule: np.ndarray,
        target_hmc_acceptance: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hmc_epsilon_schedule = np.asarray(hmc_epsilon_schedule).copy()
        self.target_hmc_acceptance = float(target_hmc_acceptance)
        self.epsilon[self.hmc_mask] = self.hmc_epsilon_schedule[self.hmc_mask]
        self.log_epsilon[self.hmc_mask] = np.log(self.epsilon[self.hmc_mask])

    def _adapt(self, pcn_acceptance: np.ndarray, hmc_acceptance: np.ndarray, sweep: int) -> None:
        # The pCN update is literally the same function used by the baseline.
        adapt_pcn_shared(self, pcn_acceptance, sweep)

        # Only the HMC component has an HMC-specific target and safe interval.
        hrate = 0.055 / math.sqrt(1.0 + sweep / 30.0)
        hmc = np.isfinite(hmc_acceptance)
        self.log_epsilon[hmc] += hrate * (
            hmc_acceptance[hmc] - self.target_hmc_acceptance
        )
        self.log_epsilon = np.clip(
            self.log_epsilon, math.log(0.0015), math.log(0.80)
        )
        self.epsilon = np.exp(self.log_epsilon)

    def _full_hmc(
        self,
        z: np.ndarray,
        phi: np.ndarray,
        beta: float,
        epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Extreme pilot proposals can overflow a polynomial transport. The
        # upstream MH calculation already rejects non-finite proposal energies;
        # suppress only those expected floating-point warnings.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return super()._full_hmc(z, phi, beta, epsilon)


def curvature_scaled_epsilon(
    problem: CurvedObservationProblem,
    cfg: SamplerConfig,
    trajectory_angle: float,
    stability_factor: float,
) -> tuple[np.ndarray, float]:
    """Prior-pilot tuning: target travel while respecting local stiffness."""
    u = np.linspace(0.0, 1.0, cfg.replicas)
    betas = u**2
    lambda95 = problem.prior_gn_lambda95()
    desired_step = trajectory_angle / cfg.hmc_steps
    stability_step = stability_factor / np.sqrt(1.0 + betas * lambda95)
    epsilon = np.minimum(desired_step, stability_step)
    return epsilon, lambda95


def _torch_synchronize(device: "torch.device") -> None:
    if torch is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _decode_torch_trace(
    problem: Any,
    source_trace: "torch.Tensor",
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract observed-prefix source and data coordinates from full retained states."""
    shape = source_trace.shape
    flattened = source_trace.reshape(-1, shape[-1])
    source_active, data_active = [], []
    with torch.no_grad():
        for start in range(0, flattened.shape[0], int(batch_size)):
            batch = flattened[start : start + int(batch_size)]
            source_active.append(problem.source_active_torch(batch).cpu())
            data_active.append(problem.data_active_torch(batch).cpu())
    source_active_tensor = torch.cat(source_active).reshape(shape[:2] + (-1,))
    data_active_tensor = torch.cat(data_active).reshape(shape[:2] + (-1,))
    return (
        source_active_tensor.numpy(),
        data_active_tensor.numpy(),
    )


class TorchSPT:
    """GPU-capable fixed-output SPT for learned differentiable transports.

    This mirrors :class:`SPT`, but keeps all replica/particle states on one
    torch device.  Exactly ``retained_sweeps`` samples are returned per cold
    particle, irrespective of the computational cost of its local kernel.
    """

    def __init__(
        self,
        problem: Any,
        cfg: SamplerConfig,
        method: str,
        seed: int,
        *,
        hmc_epsilon_schedule: np.ndarray | None = None,
        target_hmc_acceptance: float = 0.75,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for TorchSPT")
        if method not in {"pcn", "full_split_hmc_mixture"}:
            raise ValueError(f"unsupported method {method!r}")
        self.problem, self.cfg, self.method = problem, cfg, method
        self.seed = int(seed)
        self.device, self.dtype = problem.device, problem.dtype
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        unit = np.linspace(0.0, 1.0, cfg.replicas)
        self.betas = unit**2
        self.betas[0], self.betas[-1] = 0.0, 1.0
        self.hmc_mask = (
            self.betas >= cfg.beta_hmc
            if method == "full_split_hmc_mixture"
            else np.zeros(cfg.replicas, dtype=bool)
        )
        self.eta = 0.92 * (1.0 - self.betas) + 0.06 * self.betas
        self.eta[0] = 0.999
        self.logit_eta = np.log(self.eta / (1.0 - self.eta))
        self.epsilon = 0.080 * (1.0 - self.betas) + 0.050 * self.betas
        if hmc_epsilon_schedule is not None:
            schedule = np.asarray(hmc_epsilon_schedule, dtype=float)
            if schedule.shape != self.epsilon.shape:
                raise ValueError("hmc epsilon schedule has the wrong shape")
            self.epsilon[self.hmc_mask] = schedule[self.hmc_mask]
        self.log_epsilon = np.log(self.epsilon)
        self.target_hmc_acceptance = float(target_hmc_acceptance)
        self.phi_evaluations = 0
        self.gradient_evaluations = 0

    def _phi(self, z: "torch.Tensor") -> "torch.Tensor":
        self.phi_evaluations += int(np.prod(z.shape[:-1]))
        with torch.no_grad():
            return self.problem.phi_torch(z)

    def _phi_grad(
        self, z: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        self.gradient_evaluations += int(np.prod(z.shape[:-1]))
        return self.problem.phi_and_grad(z)

    def _pcn(
        self,
        z: "torch.Tensor",
        phi: "torch.Tensor",
        beta: float,
        eta: float,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        noise = torch.randn(
            z.shape, generator=self.generator, device=self.device, dtype=self.dtype
        )
        proposal = math.sqrt(max(0.0, 1.0 - eta**2)) * z + eta * noise
        proposal_phi = self._phi(proposal)
        uniform = torch.rand(
            z.shape[0], generator=self.generator,
            device=self.device, dtype=self.dtype,
        )
        accepted = uniform.log() < torch.minimum(
            torch.zeros_like(phi), -float(beta) * (proposal_phi - phi)
        )
        return (
            torch.where(accepted[:, None], proposal, z),
            torch.where(accepted, proposal_phi, phi),
            accepted,
        )

    def _split_hmc(
        self,
        z: "torch.Tensor",
        phi: "torch.Tensor",
        beta: float,
        epsilon: float,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        position = z.clone()
        momentum0 = torch.randn(
            z.shape, generator=self.generator, device=self.device, dtype=self.dtype
        )
        momentum = momentum0.clone()
        proposal_phi, gradient = self._phi_grad(position)
        momentum = momentum - 0.5 * epsilon * beta * gradient
        cosine, sine = math.cos(epsilon), math.sin(epsilon)
        for leap in range(self.cfg.hmc_steps):
            old_position = position
            position = cosine * old_position + sine * momentum
            momentum = -sine * old_position + cosine * momentum
            proposal_phi, gradient = self._phi_grad(position)
            kick = 0.5 if leap == self.cfg.hmc_steps - 1 else 1.0
            momentum = momentum - kick * epsilon * beta * gradient
        current_h = 0.5 * (z.square() + momentum0.square()).sum(dim=1) + beta * phi
        proposal_h = (
            0.5 * (position.square() + momentum.square()).sum(dim=1)
            + beta * proposal_phi
        )
        uniform = torch.rand(
            z.shape[0], generator=self.generator,
            device=self.device, dtype=self.dtype,
        )
        accepted = uniform.log() < torch.minimum(
            torch.zeros_like(current_h), current_h - proposal_h
        )
        return (
            torch.where(accepted[:, None], position, z),
            torch.where(accepted, proposal_phi, phi),
            accepted,
        )

    def _adapt(
        self,
        pcn_acceptance: np.ndarray,
        hmc_acceptance: np.ndarray,
        sweep: int,
    ) -> None:
        rate = 0.10 / math.sqrt(1.0 + sweep / 30.0)
        target = np.full(self.cfg.replicas, 0.30)
        target[0] = 1.0
        finite = np.isfinite(pcn_acceptance)
        self.logit_eta[finite] += rate * (
            pcn_acceptance[finite] - target[finite]
        )
        self.logit_eta = np.clip(self.logit_eta, -7.0, 6.9)
        self.eta = 1.0 / (1.0 + np.exp(-self.logit_eta))
        self.eta[0] = 0.999

        hrate = 0.055 / math.sqrt(1.0 + sweep / 30.0)
        finite_hmc = np.isfinite(hmc_acceptance)
        self.log_epsilon[finite_hmc] += hrate * (
            hmc_acceptance[finite_hmc] - self.target_hmc_acceptance
        )
        self.log_epsilon = np.clip(
            self.log_epsilon, math.log(0.0015), math.log(0.80)
        )
        self.epsilon = np.exp(self.log_epsilon)

    def run(self) -> RunResult:
        cfg = self.cfg
        _torch_synchronize(self.device)
        start_time = time.perf_counter()
        position = torch.randn(
            cfg.replicas, cfg.particles, cfg.dim,
            generator=self.generator, device=self.device, dtype=self.dtype,
        )
        phi = self._phi(position)
        total = cfg.adapt_sweeps + cfg.burnin_sweeps + cfg.retained_sweeps * cfg.thin
        local_accept = np.zeros(cfg.replicas)
        local_total = np.zeros(cfg.replicas)
        hmc_accept = np.zeros(cfg.replicas)
        hmc_total = np.zeros(cfg.replicas)
        pcn_accept = np.zeros(cfg.replicas)
        pcn_total = np.zeros(cfg.replicas)
        swap_accept = np.zeros(max(0, cfg.replicas - 1))
        swap_total = np.zeros(max(0, cfg.replicas - 1))
        retained: list[torch.Tensor] = []
        adapt_end = start_time
        burnin_end = start_time

        for sweep in range(total):
            sweep_hmc = np.full(cfg.replicas, np.nan)
            sweep_pcn = np.full(cfg.replicas, np.nan)
            for level, beta in enumerate(self.betas):
                if self.hmc_mask[level]:
                    hmc_count = int(round(
                        cfg.particles * cfg.cold_hmc_probability
                    ))
                    use_hmc = torch.zeros(
                        cfg.particles, dtype=torch.bool, device=self.device
                    )
                    if hmc_count:
                        order = torch.randperm(
                            cfg.particles, generator=self.generator,
                            device=self.device,
                        )[:hmc_count]
                        use_hmc[order] = True
                    accepted = torch.zeros_like(use_hmc)
                    if bool(use_hmc.any()):
                        updated, updated_phi, accepted_hmc = self._split_hmc(
                            position[level, use_hmc], phi[level, use_hmc],
                            float(beta), float(self.epsilon[level]),
                        )
                        position[level, use_hmc], phi[level, use_hmc] = updated, updated_phi
                        accepted[use_hmc] = accepted_hmc
                        sweep_hmc[level] = float(accepted_hmc.float().mean().cpu())
                        if sweep >= cfg.adapt_sweeps:
                            hmc_accept[level] += int(accepted_hmc.sum().cpu())
                            hmc_total[level] += accepted_hmc.numel()
                    use_pcn = ~use_hmc
                    if bool(use_pcn.any()):
                        updated, updated_phi, accepted_pcn = self._pcn(
                            position[level, use_pcn], phi[level, use_pcn],
                            float(beta), float(self.eta[level]),
                        )
                        position[level, use_pcn], phi[level, use_pcn] = updated, updated_phi
                        accepted[use_pcn] = accepted_pcn
                        sweep_pcn[level] = float(accepted_pcn.float().mean().cpu())
                        if sweep >= cfg.adapt_sweeps:
                            pcn_accept[level] += int(accepted_pcn.sum().cpu())
                            pcn_total[level] += accepted_pcn.numel()
                else:
                    updated, updated_phi, accepted = self._pcn(
                        position[level], phi[level], float(beta),
                        float(self.eta[level]),
                    )
                    position[level], phi[level] = updated, updated_phi
                    sweep_pcn[level] = float(accepted.float().mean().cpu())
                    if sweep >= cfg.adapt_sweeps:
                        pcn_accept[level] += int(accepted.sum().cpu())
                        pcn_total[level] += accepted.numel()

                if sweep >= cfg.adapt_sweeps:
                    local_accept[level] += int(accepted.sum().cpu())
                    local_total[level] += cfg.particles

            for parity in (sweep % 2, 1 - sweep % 2):
                for level in range(parity, cfg.replicas - 1, 2):
                    log_ratio = (self.betas[level + 1] - self.betas[level]) * (
                        phi[level + 1] - phi[level]
                    )
                    uniform = torch.rand(
                        cfg.particles, generator=self.generator,
                        device=self.device, dtype=self.dtype,
                    )
                    accepted_swap = uniform.log() < torch.minimum(
                        torch.zeros_like(log_ratio), log_ratio
                    )
                    if bool(accepted_swap.any()):
                        lower = position[level, accepted_swap].clone()
                        position[level, accepted_swap] = position[level + 1, accepted_swap]
                        position[level + 1, accepted_swap] = lower
                        lower_phi = phi[level, accepted_swap].clone()
                        phi[level, accepted_swap] = phi[level + 1, accepted_swap]
                        phi[level + 1, accepted_swap] = lower_phi
                    if sweep >= cfg.adapt_sweeps:
                        swap_accept[level] += int(accepted_swap.sum().cpu())
                        swap_total[level] += cfg.particles

            if sweep < cfg.adapt_sweeps:
                self._adapt(sweep_pcn, sweep_hmc, sweep)
                if sweep == cfg.adapt_sweeps - 1:
                    _torch_synchronize(self.device)
                    adapt_end = time.perf_counter()
                continue
            retained_index = sweep - cfg.adapt_sweeps - cfg.burnin_sweeps
            if retained_index == -1:
                _torch_synchronize(self.device)
                burnin_end = time.perf_counter()
            if retained_index >= 0 and retained_index % cfg.thin == 0:
                # On CPU, detach().cpu() still aliases the in-place replica state.
                retained.append(position[-1].detach().cpu().clone())

        _torch_synchronize(self.device)
        sampling_end = time.perf_counter()
        source_trace = torch.stack(retained).to(
            device=self.device, dtype=self.dtype
        )
        source_active, data_active = _decode_torch_trace(
            self.problem, source_trace
        )
        _torch_synchronize(self.device)
        decode_end = time.perf_counter()
        hmc_acceptance = np.divide(
            hmc_accept, hmc_total, out=np.full_like(hmc_accept, np.nan),
            where=hmc_total > 0,
        )
        pcn_acceptance = np.divide(
            pcn_accept, pcn_total, out=np.full_like(pcn_accept, np.nan),
            where=pcn_total > 0,
        )
        return RunResult(
            method=self.method,
            scenario=self.problem.scenario,
            seed=self.seed,
            active_trace=source_active,
            data_active_trace=data_active,
            local_acceptance=local_accept / np.maximum(local_total, 1.0),
            swap_acceptance=np.divide(
                swap_accept, swap_total,
                out=np.full_like(swap_accept, np.nan), where=swap_total > 0,
            ),
            proposal_scale=np.where(self.hmc_mask, self.epsilon, self.eta),
            runtime_sec=decode_end - start_time,
            phi_evaluations=self.phi_evaluations,
            gradient_evaluations=self.gradient_evaluations,
            hmc_mask=self.hmc_mask.copy(),
            hmc_acceptance=hmc_acceptance,
            pcn_acceptance=pcn_acceptance,
            hmc_move_fraction=np.divide(
                hmc_total, hmc_total + pcn_total,
                out=np.zeros_like(hmc_total),
                where=(hmc_total + pcn_total) > 0,
            ),
            timing_sec={
                "adaptation": max(0.0, adapt_end - start_time),
                "burnin_after_adaptation": max(0.0, burnin_end - adapt_end),
                "retained_sampling": max(0.0, sampling_end - burnin_end),
                "final_transport": max(0.0, decode_end - sampling_end),
                "total": decode_end - start_time,
            },
            retained_per_chain=cfg.retained_sweeps,
            retained_chains=cfg.particles,
        )


class TorchClassicalHMC:
    """Published-SGFM-style classical HMC without parallel tempering."""

    def __init__(
        self,
        problem: Any,
        cfg: SamplerConfig,
        seed: int,
        *,
        initial_epsilon: float,
        target_acceptance: float = 0.60,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for TorchClassicalHMC")
        self.problem, self.cfg, self.seed = problem, cfg, int(seed)
        self.device, self.dtype = problem.device, problem.dtype
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.log_epsilon = math.log(float(initial_epsilon))
        self.target_acceptance = float(target_acceptance)
        self.phi_evaluations = 0
        self.gradient_evaluations = 0

    def _phi(self, z: "torch.Tensor") -> "torch.Tensor":
        self.phi_evaluations += z.shape[0]
        with torch.no_grad():
            return self.problem.phi_torch(z)

    def _phi_grad(self, z: "torch.Tensor"):
        self.gradient_evaluations += z.shape[0]
        return self.problem.phi_and_grad(z)

    def _proposal(self, z: "torch.Tensor", phi: "torch.Tensor"):
        epsilon = math.exp(self.log_epsilon)
        position = z.clone()
        momentum0 = torch.randn(
            z.shape, generator=self.generator, device=self.device, dtype=self.dtype
        )
        momentum = momentum0.clone()
        proposal_phi, likelihood_gradient = self._phi_grad(position)
        momentum -= 0.5 * epsilon * (position + likelihood_gradient)
        for leap in range(self.cfg.hmc_steps):
            position = position + epsilon * momentum
            proposal_phi, likelihood_gradient = self._phi_grad(position)
            kick = 0.5 if leap == self.cfg.hmc_steps - 1 else 1.0
            momentum -= kick * epsilon * (position + likelihood_gradient)
        current_h = 0.5 * (z.square() + momentum0.square()).sum(dim=1) + phi
        proposal_h = 0.5 * (
            position.square() + momentum.square()
        ).sum(dim=1) + proposal_phi
        uniform = torch.rand(
            z.shape[0], generator=self.generator,
            device=self.device, dtype=self.dtype,
        )
        accepted = uniform.log() < torch.minimum(
            torch.zeros_like(current_h), current_h - proposal_h
        )
        return (
            torch.where(accepted[:, None], position, z),
            torch.where(accepted, proposal_phi, phi),
            accepted,
        )

    def run(self) -> RunResult:
        cfg = self.cfg
        _torch_synchronize(self.device)
        start_time = time.perf_counter()
        position = torch.randn(
            cfg.particles, cfg.dim, generator=self.generator,
            device=self.device, dtype=self.dtype,
        )
        phi = self._phi(position)
        total = cfg.adapt_sweeps + cfg.burnin_sweeps + cfg.retained_sweeps * cfg.thin
        retained: list[torch.Tensor] = []
        accepted_after_adaptation = 0
        proposals_after_adaptation = 0
        adapt_end = start_time
        burnin_end = start_time
        for sweep in range(total):
            position, phi, accepted = self._proposal(position, phi)
            acceptance = float(accepted.float().mean().cpu())
            if sweep < cfg.adapt_sweeps:
                rate = 0.055 / math.sqrt(1.0 + sweep / 30.0)
                self.log_epsilon += rate * (
                    acceptance - self.target_acceptance
                )
                self.log_epsilon = float(np.clip(
                    self.log_epsilon, math.log(1e-5), math.log(0.80)
                ))
                if sweep == cfg.adapt_sweeps - 1:
                    _torch_synchronize(self.device)
                    adapt_end = time.perf_counter()
                continue
            accepted_after_adaptation += int(accepted.sum().cpu())
            proposals_after_adaptation += accepted.numel()
            retained_index = sweep - cfg.adapt_sweeps - cfg.burnin_sweeps
            if retained_index == -1:
                _torch_synchronize(self.device)
                burnin_end = time.perf_counter()
            if retained_index >= 0 and retained_index % cfg.thin == 0:
                retained.append(position.detach().cpu())
        _torch_synchronize(self.device)
        sampling_end = time.perf_counter()
        source_trace = torch.stack(retained).to(self.device, self.dtype)
        source_active, data_active = _decode_torch_trace(
            self.problem, source_trace
        )
        _torch_synchronize(self.device)
        decode_end = time.perf_counter()
        acceptance = accepted_after_adaptation / max(proposals_after_adaptation, 1)
        return RunResult(
            method="sgfm_classical_hmc",
            scenario=self.problem.scenario,
            seed=self.seed,
            active_trace=source_active,
            data_active_trace=data_active,
            local_acceptance=np.asarray([acceptance]),
            swap_acceptance=np.empty(0),
            proposal_scale=np.asarray([math.exp(self.log_epsilon)]),
            runtime_sec=decode_end - start_time,
            phi_evaluations=self.phi_evaluations,
            gradient_evaluations=self.gradient_evaluations,
            hmc_mask=np.asarray([True]),
            hmc_acceptance=np.asarray([acceptance]),
            pcn_acceptance=np.asarray([np.nan]),
            hmc_move_fraction=np.asarray([1.0]),
            timing_sec={
                "adaptation": max(0.0, adapt_end - start_time),
                "burnin_after_adaptation": max(0.0, burnin_end - adapt_end),
                "retained_sampling": max(0.0, sampling_end - burnin_end),
                "final_transport": max(0.0, decode_end - sampling_end),
                "total": decode_end - start_time,
            },
            retained_per_chain=cfg.retained_sweeps,
            retained_chains=cfg.particles,
        )
