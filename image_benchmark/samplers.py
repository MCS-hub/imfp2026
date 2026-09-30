"""Image-scale execution of the supplied SPT-pCN / split-HMC hybrid kernels."""
import math
import time
import numpy as np
import torch
from .vendor import enable_vendor
from .schedule import SweepSchedule, PHASES

enable_vendor()
from spt_score import SamplerConfig, TorchSPT


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def split_trajectory(position, momentum, beta, epsilon, steps, phi_grad):
    """Likelihood kicks surrounding exact rotations of the Gaussian Hamiltonian."""
    value, gradient = phi_grad(position)
    momentum = momentum - 0.5 * epsilon * beta * gradient
    cosine, sine = math.cos(epsilon), math.sin(epsilon)
    for leap in range(steps):
        old_position = position
        position = cosine * old_position + sine * momentum
        momentum = -sine * old_position + cosine * momentum
        value, gradient = phi_grad(position)
        kick = 0.5 if leap == steps - 1 else 1.0
        momentum = momentum - kick * epsilon * beta * gradient
    return position, momentum, value


def gaussian_energy(position, momentum):
    return 0.5 * (position.double().square() + momentum.double().square()).sum(-1)


def swap_log_ratio(beta_low, beta_high, phi_low, phi_high):
    return (beta_high - beta_low) * (phi_high - phi_low)


class ImageSPT(TorchSPT):
    """Same target and schedule as TorchSPT, with microbatching and CPU traces.

    Pixel likelihoods can require steps below the synthetic benchmark's
    0.0015 floor. All adaptation is restricted to the declared warmup.
    """
    def __init__(self, problem, config, method, seed):
        cfg = SamplerConfig(dim=problem.dim, replicas=config["replicas"],
                            particles=config["chains"], adapt_sweeps=config["adapt_sweeps"],
                            burnin_sweeps=config["burnin_sweeps"],
                            retained_sweeps=config["retained_per_chain"], thin=config["thin"],
                            beta_hmc=1.0, hmc_steps=config["hmc_steps"],
                            cold_hmc_probability=1.0)
        super().__init__(problem, cfg, "pcn" if method in ("imf_spt_pcn", "sit_spt_pcn")
                         else "full_split_hmc_mixture", seed,
                         hmc_epsilon_schedule=np.full(cfg.replicas, config["initial_hmc_epsilon"]),
                         target_hmc_acceptance=config["target_hmc_acceptance"])
        self.settings = config
        if config.get("betas") is not None:
            self.betas = np.asarray(config["betas"], dtype=float)
        else:
            self.betas = np.linspace(0, 1, cfg.replicas)**config["temperature_power"]
        self.hmc_mask = np.zeros(cfg.replicas, dtype=bool)
        self.hmc_mask[-1] = method == "imf_spt_hybrid"
        self.eta = 0.92 * (1-self.betas) + config["initial_pcn_scale"] * self.betas
        self.eta[0] = 0.999
        self.logit_eta = np.log(self.eta / (1-self.eta))

    def _pcn(self, z, phi, beta, eta):
        noise = torch.randn(z.shape, generator=self.generator, device=self.device, dtype=self.dtype)
        proposal = math.sqrt(1-eta**2) * z + eta * noise
        proposal_phi = self._phi(proposal)
        ratio = -beta * (proposal_phi - phi)
        uniform = torch.rand(len(z), generator=self.generator, device=self.device, dtype=self.dtype)
        accepted = torch.isfinite(proposal_phi) & (uniform.log() < ratio.clamp(max=0))
        return torch.where(accepted[:, None], proposal, z), torch.where(accepted, proposal_phi, phi), accepted

    def _split_hmc(self, z, phi, beta, epsilon):
        momentum = torch.randn(z.shape, generator=self.generator, device=self.device, dtype=self.dtype)
        proposal, final_momentum, proposal_phi = split_trajectory(
            z.clone(), momentum.clone(), beta, epsilon, self.cfg.hmc_steps, self._phi_grad)
        delta = gaussian_energy(z, momentum) + beta * phi - (
            gaussian_energy(proposal, final_momentum) + beta * proposal_phi)
        uniform = torch.rand(len(z), generator=self.generator, device=self.device, dtype=self.dtype)
        accepted = torch.isfinite(delta) & (uniform.log() < delta.clamp(max=0))
        return torch.where(accepted[:, None], proposal, z), torch.where(accepted, proposal_phi, phi), accepted

    def _adapt(self, pcn_acceptance, hmc_acceptance, sweep):
        rate = 0.10 / math.sqrt(1 + sweep / 30)
        target = np.full(self.cfg.replicas, self.settings["target_pcn_acceptance"])
        target[0] = 1
        finite = np.isfinite(pcn_acceptance)
        self.logit_eta[finite] += rate * (pcn_acceptance[finite] - target[finite])
        self.logit_eta = np.clip(self.logit_eta, -13.8, 6.9)
        self.eta = 1 / (1 + np.exp(-self.logit_eta))
        self.eta[0] = 0.999
        finite = np.isfinite(hmc_acceptance)
        rate_hmc = 0.055 / math.sqrt(1 + sweep / 30)
        self.log_epsilon[finite] += rate_hmc * (hmc_acceptance[finite] - self.target_hmc_acceptance)
        self.log_epsilon = np.clip(self.log_epsilon, math.log(self.settings["min_hmc_epsilon"]),
                                  math.log(self.settings["max_hmc_epsilon"]))
        self.epsilon = np.exp(self.log_epsilon)

    def run(self, progress=None, diagnostics=False, sweep_callback=None, time_budget=None):
        cfg = self.cfg
        synchronize(self.device)
        start = time.perf_counter()
        z = torch.randn(cfg.replicas, cfg.particles, cfg.dim, generator=self.generator,
                        device=self.device, dtype=self.dtype)
        phi = self._phi(z)
        if not torch.isfinite(phi).all():
            raise FloatingPointError("Nonfinite initial likelihood; check model and observation units")
        synchronize(self.device)
        init_end = time.perf_counter()
        local_hits = np.zeros(cfg.replicas)
        local_n = np.zeros(cfg.replicas)
        swap_hits = np.zeros(cfg.replicas-1)
        swap_n = np.zeros(cfg.replicas-1)
        traces, likelihoods = [], []
        sweep_rows = []
        phase_seconds = dict.fromkeys(PHASES, 0.)
        if diagnostics:
            initial_cold = z[-1].detach().cpu().clone().numpy()
            local_by_chain = np.zeros((cfg.replicas, cfg.particles), dtype=np.int64)
            swap_by_chain = np.zeros((cfg.replicas-1, cfg.particles), dtype=np.int64)
            labels = np.repeat(np.arange(cfg.replicas)[:, None], cfg.particles, axis=1)
            potential_history, label_history = [], []
        total = cfg.adapt_sweeps + cfg.burnin_sweeps + cfg.retained_sweeps * cfg.thin
        budget_sec = None if time_budget is None else time_budget['sampling_sec']
        # Initialization is charged to the sampling budget before allocating phases.
        available = None if budget_sec is None else max(0., budget_sec-(init_end-start))
        schedule = SweepSchedule((cfg.adapt_sweeps,cfg.burnin_sweeps,cfg.retained_sweeps*cfg.thin),
                                 available, (1/6,1/6,2/3) if time_budget is None else time_budget['phase_fractions'])
        sweep = 0
        while True:
            synchronize(self.device)
            now = time.perf_counter()
            phase = schedule.next_phase(now-init_end)
            if phase is None: break
            if time_budget is not None and len(traces) >= 8:
                estimate = max(schedule.durations[-5:], default=0.)
                # Rendering all retained cold states costs roughly one replica's
                # share of each sweep. Include headroom for PNG export/diagnostics.
                output_estimate = 1.5 * estimate/cfg.replicas * (len(traces)+1) + 120
                total_limit = time_budget.get('remaining_total_sec', budget_sec+3600)
                if now-start + 1.1*estimate + output_estimate >= total_limit:
                    break
            sweep_start = now
            pcn_acceptance = np.full(cfg.replicas, np.nan)
            hmc_acceptance = np.full(cfg.replicas, np.nan)
            for k, beta in enumerate(self.betas):
                if self.hmc_mask[k]:
                    z[k], phi[k], accepted = self._split_hmc(z[k], phi[k], float(beta), self.epsilon[k])
                    hmc_acceptance[k] = float(accepted.double().mean())
                else:
                    z[k], phi[k], accepted = self._pcn(z[k], phi[k], float(beta), self.eta[k])
                    pcn_acceptance[k] = float(accepted.double().mean())
                if phase != 'adaptation':
                    local_hits[k] += int(accepted.sum())
                    local_n[k] += cfg.particles
                    if diagnostics:
                        local_by_chain[k] += accepted.cpu().numpy()
            # Match both alternating parity passes of the supplied synthetic code.
            for parity in (sweep % 2, 1-sweep % 2):
                for k in range(parity, cfg.replicas-1, 2):
                    ratio = swap_log_ratio(self.betas[k], self.betas[k+1], phi[k], phi[k+1])
                    uniform = torch.rand(cfg.particles, generator=self.generator,
                                         device=self.device, dtype=self.dtype)
                    accept = uniform.log() < ratio.clamp(max=0)
                    low_z, low_phi = z[k].clone(), phi[k].clone()
                    z[k] = torch.where(accept[:, None], z[k+1], z[k])
                    phi[k] = torch.where(accept, phi[k+1], phi[k])
                    z[k+1] = torch.where(accept[:, None], low_z, z[k+1])
                    phi[k+1] = torch.where(accept, low_phi, phi[k+1])
                    if diagnostics:
                        flags = accept.cpu().numpy()
                        lo, hi = labels[k].copy(), labels[k+1].copy()
                        labels[k] = np.where(flags, hi, lo)
                        labels[k+1] = np.where(flags, lo, hi)
                        if phase != 'adaptation':
                            swap_by_chain[k] += flags
                    if phase != 'adaptation':
                        swap_hits[k] += int(accept.sum())
                        swap_n[k] += cfg.particles
            if diagnostics:
                potential_history.append(phi.detach().cpu().numpy().copy())
                label_history.append(labels.copy())
            if phase == 'adaptation':
                self._adapt(pcn_acceptance, hmc_acceptance, schedule.counts[phase])
            if phase == 'retained_sampling' and schedule.counts[phase] % cfg.thin == 0:
                # clone is essential on CPU, where .cpu() alone aliases the state.
                traces.append(z[-1].detach().cpu().clone())
                likelihoods.append(phi[-1].detach().cpu().clone())
            synchronize(self.device)
            sweep_end = time.perf_counter()
            seconds = sweep_end-sweep_start
            schedule.record(phase, seconds)
            phase_seconds[phase] += seconds
            sweep += 1
            row = {'sweep':sweep,'phase':phase,'phase_sweep':schedule.counts[phase],
                   'sweep_sec':seconds,'elapsed_sampling_sec':sweep_end-start,
                   'cold_phi_mean':float(phi[-1].mean()),'retained_draws':len(traces)}
            sweep_rows.append(row)
            if sweep_callback: sweep_callback(row)
            if progress and (sweep % self.settings['log_every'] == 0 or (budget_sec is None and sweep == total)):
                limit = total if budget_sec is None else 'time budget'
                progress(f"sweep {sweep}/{limit}; phase={phase}; sweep_sec={seconds:.3f}; "
                         f"cold Phi mean={float(phi[-1].mean()):.3g}; "
                         f"cold scale={self.epsilon[-1] if self.hmc_mask[-1] else self.eta[-1]:.3g}")
        synchronize(self.device)
        finish = time.perf_counter()
        if not traces:
            raise RuntimeError('Sampling budget exhausted before any retained draw; increase budget')
        result = {
            'phase_sweeps':schedule.counts, 'sweep_timings':sweep_rows,
            'sweep_sec_mean':float(np.mean(schedule.durations)),
            'sweep_sec_median':float(np.median(schedule.durations)),
            'sweep_sec_by_phase':{name:float(np.mean([r['sweep_sec'] for r in sweep_rows if r['phase']==name]))
                                 if schedule.counts[name] else None for name in PHASES},
            'sampling_budget_sec':budget_sec,

            "source": torch.stack(traces), "phi": torch.stack(likelihoods).numpy(),
            "sampling_sec": finish-start,
            "timing": {"initialization": init_end-start, **phase_seconds},
            "kernel": {"betas": self.betas.tolist(), "pcn_scales": self.eta.tolist(),
                       "hmc_epsilon": self.epsilon.tolist(), "hmc_mask": self.hmc_mask.tolist(),
                       "local_acceptance": (local_hits / np.maximum(local_n, 1)).tolist(),
                       "swap_acceptance": (swap_hits / np.maximum(swap_n, 1)).tolist(),
                       "phi_evaluations": self.phi_evaluations,
                       "gradient_evaluations": self.gradient_evaluations}
        }

        if diagnostics:
            count = max(schedule.counts['burnin']+schedule.counts['retained_sampling'], 1)
            result["telemetry"] = {
                "initial_cold_source": initial_cold,
                "sweep_sec": np.array([r['sweep_sec'] for r in sweep_rows]),
                "elapsed_sampling_sec": np.array([r['elapsed_sampling_sec'] for r in sweep_rows]),
                "phase": np.array([r['phase'] for r in sweep_rows]),
                "potential": np.stack(potential_history),
                "replica_labels": np.stack(label_history),
                "local_acceptance_by_chain": local_by_chain/count,
                "swap_acceptance_by_chain": swap_by_chain/count,
            }
        return result
