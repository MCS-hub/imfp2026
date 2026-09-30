#!/usr/bin/env python3
"""Fixed-output end-to-end comparison with sample-only curved priors.

The experiment trains learned priors with reported parameter counts from one fixed sample
dataset, constructs the same analytic likelihood for every method, then stops
each MCMC method after exactly N retained samples per output chain.  Runtime is
reported rather than synthetically equating unlike transport-gradient calls.

Default method labels
---------------------
``imf_spt_pcn``
    Improved MeanFlow map; pCN at every SPT temperature.
``imf_spt_hybrid``
    Improved MeanFlow map; pCN at hot levels and Gaussian-reference split HMC
    at levels selected by ``--hmc-beta``.
``fm_spt_pcn``
    OT-coupled flow-matching map; pCN at every SPT temperature.
``fm_sgfm_hmc``
    Published-SGFM-style classical Euclidean HMC without tempering.
``fm_split_hmc``
    Stronger SGFM ablation: the same FM map with Gaussian-reference split HMC
    and no tempering.
``diffusion_dps``
    DDPM epsilon prior with likelihood-guided DPS.  This is approximate and is
    not an invariant MCMC kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from curved_problem import (
    CurvedObservationProblem,
    DEFAULT_BANANA_TAU, DEFAULT_SINE_TAU, GEOMETRY_VERSION,
    LearnedTransportPosterior,
    make_active_reference,
    make_datasets,
    make_pair_reference,
)
from evaluation import (
    fixed_output_metrics,
    make_end_to_end_sample_plot,
    make_reference_bins,
    write_csv,
    write_end_to_end_report,
    make_prior_diagnostic_plots,
    make_posterior_heatmap,
)
from learned_models import (
    LearnedTransport,
    MODEL_IMPLEMENTATION_VERSION,
    NetworkConfig,
    TrainingConfig,
    build_model,
    dps_sample,
    load_checkpoint,
    parameter_count,
    sample_diffusion_prior,
    sample_learned_prior,
    save_checkpoint,
    train_learned_prior,
)
from spt_score import (
    RunResult,
    SamplerConfig,
    TorchClassicalHMC,
    TorchSPT,
)


METHODS = (
    "imf_spt_pcn",
    "imf_spt_hybrid",
    "fm_spt_pcn",
    "diffusion_dps",
)


def csv_tuple(text: str, cast=str):
    return tuple(cast(item.strip()) for item in text.split(",") if item.strip())


def torch_dtype(name: str):
    return {"float32": torch.float32, "float64": torch.float64}[name]


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_kinds(methods: tuple[str, ...]):
    kinds = []
    if any(method.startswith("imf_") for method in methods):
        kinds.append("imf")
    if any(method.startswith("fm_") for method in methods):
        kinds.append("fm")
    if "diffusion_dps" in methods:
        kinds.append("diffusion")
    return tuple(kinds)


def training_config_signature(cfg: TrainingConfig, kind: str):
    """Return only training fields that affect the requested model family.

    TrainingConfig is shared across iMF, FM, and diffusion, so comparing the
    full dataclass would incorrectly invalidate (for example) an FM checkpoint
    when only an iMF-specific option changes.
    """
    common_fields = (
        "steps",
        "batch_size",
    )
    family_fields = {
        "imf": (
            "imf_coupling",
            "imf_auxiliary_weight",
            "imf_learning_rate",
            "imf_adam_beta1",
            "imf_adam_beta2",
            "imf_weight_decay",
            "imf_grad_clip",
            "imf_time_distribution",
            "imf_time_mean",
            "imf_time_std",
            "imf_equal_time_fraction",
        ),
        "fm": (
            "grad_clip",
            "learning_rate",
            "weight_decay",
            "fm_coupling",
        ),
        "diffusion": (
            "grad_clip",
            "learning_rate",
            "weight_decay",
            "diffusion_steps",
            "beta_start",
            "beta_end",
        ),
    }
    if kind not in family_fields:
        raise ValueError(f"unknown learned-prior kind {kind!r}")
    fields = common_fields + family_fields[kind]
    return {field: getattr(cfg, field) for field in fields}


def training_config_differences(
    loaded: TrainingConfig,
    current: TrainingConfig,
    kind: str,
):
    """Return model-relevant checkpoint differences as saved/current pairs."""
    saved = training_config_signature(loaded, kind)
    wanted = training_config_signature(current, kind)
    return {
        key: (saved[key], wanted[key])
        for key in wanted
        if saved[key] != wanted[key]
    }


def rectangularize_rows(rows: list[dict[str, object]]):
    """Return CSV-safe rows whose keys share one stable union schema."""
    fieldnames = list(dict.fromkeys(
        key
        for row in rows
        for key in row
    ))
    return [
        {fieldname: row.get(fieldname, "") for fieldname in fieldnames}
        for row in rows
    ]



def dps_result(
    samples: np.ndarray,
    problem: CurvedObservationProblem,
    *,
    retained_per_chain: int,
    chains: int,
    seed: int,
    timing: dict[str, float],
    diffusion_steps: int,
):
    active = problem.data_active(samples).reshape(
        retained_per_chain, chains, problem.active_dim
    )
    return RunResult(
        method="diffusion_dps",
        scenario=problem.scenario,
        seed=seed,
        active_trace=active.copy(),
        data_active_trace=active,
        local_acceptance=np.asarray([np.nan]),
        swap_acceptance=np.empty(0),
        proposal_scale=np.asarray([timing["guidance_scale"]]),
        runtime_sec=float(timing["runtime_sec"]),
        phi_evaluations=0,
        gradient_evaluations=int(samples.shape[0] * diffusion_steps),
        hmc_mask=np.asarray([False]),
        hmc_acceptance=np.asarray([np.nan]),
        pcn_acceptance=np.asarray([np.nan]),
        hmc_move_fraction=np.asarray([0.0]),
        timing_sec={
            "adaptation": 0.0,
            "burnin_after_adaptation": 0.0,
            "retained_sampling": float(timing["runtime_sec"]),
            "final_transport": 0.0,
            "total": float(timing["runtime_sec"]),
        },
        retained_per_chain=retained_per_chain,
        retained_chains=chains,
    )


def apply_sampling_reporting(row: dict[str, object]) -> None:
    """Keep throughput for every method; omit artificial DPS chain diagnostics."""
    runtime = float(row["runtime_sec"])
    row["samples_per_sec"] = (
        float(row["nominal_retained_samples"]) / runtime
        if np.isfinite(runtime) and runtime > 0.0 else float("nan")
    )
    if row["method"] == "diffusion_dps":
        # DPS outputs are independent diffusion trajectories. Their reshape into
        # chains is only a compatibility device for the shared metric routine.
        for key in row:
            if "ess" in key.split("_") or "rhat" in key.split("_"):
                row[key] = float("nan")
        for key in (
            "cold_acceptance", "cold_hmc_acceptance", "cold_pcn_acceptance",
            "minimum_swap_acceptance",
        ):
            if key in row:
                row[key] = float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outs/end_to_end_results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_end_to_end"))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--scenario", choices=("banana", "sine"), default="banana")
    parser.add_argument("--banana-tau", type=float, default=DEFAULT_BANANA_TAU)
    parser.add_argument("--sine-tau", type=float, default=DEFAULT_SINE_TAU)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--observed-blocks", type=int, default=16)
    parser.add_argument("--sigma-y", type=float, default=0.2)
    parser.add_argument(
        "--observation-tilt", type=float, default=0.35,
        help="c in the data-space operator G(x1,x2)=x2+c*x1",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("fixed", "prior-predictive"),
        default="fixed",
        help="fixed gives a controlled unequal-weight target; prior-predictive generates y",
    )
    parser.add_argument(
        "--leading-observation", "--fixed-observation",
        dest="leading_observation", type=float, default=2.0,
        help="observation assigned to the leading blocks in fixed mode",
    )
    parser.add_argument(
        "--leading-blocks", "--multimodal-blocks",
        dest="leading_blocks", type=int, default=2,
        help="number of leading blocks assigned --leading-observation",
    )
    parser.add_argument(
        "--remaining-observation", "--unimodal-observation",
        dest="remaining_observation", type=float, default=-1.0,
        help="observation assigned to all remaining blocks in fixed mode",
    )
    parser.add_argument("--observation-seed", type=int, default=29)
    parser.add_argument("--generated-noise-sigma", type=float, default=0.0)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--network-hidden", type=int, default=512)
    parser.add_argument("--network-depth", type=int, default=4)
    parser.add_argument("--time-harmonics", type=int, default=4,
                        help="time Fourier features for FM/diffusion; iMF uses raw r,t")
    parser.add_argument("--imf-network-depth", type=int, default=4,
                        help="iMF hidden layers")
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="FM/diffusion gradient clipping threshold; 0 disables")
    parser.add_argument("--imf-grad-clip", type=float, default=0.0,
                        help="iMF-only gradient clipping; disabled in the working recipe")
    parser.add_argument("--imf-weight-decay", type=float, default=1e-5,
                        help="Adam weight decay for iMF")
    parser.add_argument("--imf-time-distribution", default="uniform",
                        choices=("uniform", "sorted-uniform", "logit-normal"),
                        help="uniform means independent unsorted r,t; other choices are ablations")
    parser.add_argument("--training-samples", type=int, default=20_000)
    parser.add_argument("--training-steps", type=int, default=16_000)
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--training-seed", type=int, default=1701)
    parser.add_argument(
        "--imf-coupling",
        choices=("independent", "minibatch-ot"),
        default="independent",
        help=(
            "data/noise coupling for iMF training; independent matches the "
            "theory in this study, minibatch-ot uses the same hard minibatch "
            "OT pairing available to flow matching"
        ),
    )
    parser.add_argument(
        "--imf-auxiliary-weight", type=float, default=1.0,
        help="lambda multiplying the auxiliary velocity risk",
    )
    parser.add_argument(
        "--imf-learning-rate", type=float, default=1e-3,
        help="iMF-specific learning rate; supplied auxiliary-head recipe uses 1e-3",
    )
    parser.add_argument("--imf-adam-beta1", type=float, default=0.9)
    parser.add_argument("--imf-adam-beta2", type=float, default=0.999)
    parser.add_argument(
        "--imf-time-mean", type=float, default=-0.4,
        help="mean of the Gaussian logits for the iMF logit-normal time law",
    )
    parser.add_argument(
        "--imf-time-std", type=float, default=1.0,
        help="standard deviation of Gaussian logits for the iMF logit-normal time law",
    )
    parser.add_argument(
        "--imf-equal-time-fraction", type=float, default=0.0,
        help="optional fraction collapsed to r=t; working recipe uses 0",
    )
    parser.add_argument(
        "--fm-coupling",
        choices=("independent", "minibatch-ot"),
        default="independent",
    )
    parser.add_argument("--reuse-checkpoints", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--imf-steps", type=int, default=6)
    parser.add_argument(
        "--fm-steps", type=int, default=100,
        help="fixed ODE macro-steps; SGFM reports a two-step Dopri5 map",
    )
    parser.add_argument(
        "--fm-solver", choices=("euler", "midpoint", "rk4", "dopri5"), default="rk4",
        help="use dopri5 for the closest SGFM solver match (requires torchdiffeq)",
    )
    parser.add_argument("--fm-rtol", type=float, default=1e-5)
    parser.add_argument("--fm-atol", type=float, default=1e-5)
    parser.add_argument("--replicas", type=int, default=24)
    parser.add_argument("--chains", type=int, default=10)
    parser.add_argument("--adapt-sweeps", type=int, default=500)
    parser.add_argument("--burnin-sweeps", type=int, default=500)
    parser.add_argument("--retained-per-chain", type=int, default=600)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--hmc-beta", type=float, default=1.0)
    parser.add_argument("--hmc-steps", type=int, default=6)
    parser.add_argument("--cold-hmc-probability", type=float, default=1.0)
    parser.add_argument("--initial-hmc-epsilon", type=float, default=0.05)
    parser.add_argument("--target-split-hmc-acceptance", type=float, default=0.75)
    parser.add_argument("--target-sgfm-hmc-acceptance", type=float, default=0.60)
    parser.add_argument("--sampler-seeds", default="0,1,2,3")

    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--dps-guidance-scale", type=float, default=0.3)
    parser.add_argument("--dps-batch-size", type=int, default=256)

    parser.add_argument("--reference-grid-n", type=int, default=501)
    parser.add_argument("--reference-samples", type=int, default=50_000)
    parser.add_argument("--sliced-directions", type=int, default=48)
    parser.add_argument("--prior-diagnostic-samples", type=int, default=10_000)
    parser.add_argument("--plot-points", type=int, default=2500)
    parser.add_argument("--heatmap-bins", type=int, default=64,
                        help="bins per axis for sample heatmaps; all samples and a shared density scale per figure")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.heatmap_bins < 2:
        parser.error("--heatmap-bins must be at least 2")

    methods = csv_tuple(args.methods)
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if not methods:
        raise ValueError("--methods must contain at least one method")
    if args.dim < 2 or args.dim % 2:
        raise ValueError("dim must be positive and even; all dim/2 blocks are curved")
    if not 1 <= args.observed_blocks <= args.dim // 2:
        raise ValueError("observed-blocks must lie in [1, dim/2]")
    if not 0 <= args.leading_blocks <= args.observed_blocks:
        raise ValueError("leading-blocks must lie in [0, observed-blocks]")
    if args.retained_per_chain < 4:
        raise ValueError("retained_per_chain must be at least 4")
    if args.chains < 2:
        raise ValueError("at least two output chains are required for split-Rhat")
    if not 0.0 <= args.hmc_beta <= 1.0:
        raise ValueError("--hmc-beta must lie in [0, 1]")
    if not 0.0 <= args.cold_hmc_probability <= 1.0:
        raise ValueError("--cold-hmc-probability must lie in [0, 1]")
    if args.imf_network_depth < 1:
        raise ValueError("--imf-network-depth must be positive")
    if min(args.grad_clip, args.imf_grad_clip, args.imf_weight_decay) < 0:
        raise ValueError("gradient clipping and weight decay must be nonnegative")
    if args.imf_auxiliary_weight <= 0.0:
        raise ValueError("--imf-auxiliary-weight must be positive")
    if args.imf_learning_rate <= 0.0:
        raise ValueError("--imf-learning-rate must be positive")
    if not 0.0 < args.imf_adam_beta1 < 1.0:
        raise ValueError("--imf-adam-beta1 must lie in (0, 1)")
    if not 0.0 < args.imf_adam_beta2 < 1.0:
        raise ValueError("--imf-adam-beta2 must lie in (0, 1)")
    if args.imf_time_std <= 0.0:
        raise ValueError("--imf-time-std must be positive")
    if not 0.0 <= args.imf_equal_time_fraction <= 1.0:
        raise ValueError("--imf-equal-time-fraction must lie in [0, 1]")
    if args.quick:
        args.training_samples = min(args.training_samples, 1024)
        args.training_steps = min(args.training_steps, 8)
        args.training_batch_size = min(args.training_batch_size, 64)
        args.adapt_sweeps = min(args.adapt_sweeps, 3)
        args.burnin_sweeps = min(args.burnin_sweeps, 3)
        args.retained_per_chain = min(args.retained_per_chain, 6)
        args.replicas = min(args.replicas, 4)
        args.chains = min(args.chains, 3)
        args.hmc_steps = min(args.hmc_steps, 2)
        args.fm_steps = min(args.fm_steps, 2)
        args.diffusion_steps = min(args.diffusion_steps, 8)
        args.reference_grid_n = min(args.reference_grid_n, 151)
        args.reference_samples = min(args.reference_samples, 3000)
        args.prior_diagnostic_samples = min(args.prior_diagnostic_samples, 1000)
        args.sampler_seeds = "0"

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm was requested but torch.cuda.is_available() is false")
    torch.manual_seed(args.training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.training_seed)

    sampler_cfg = SamplerConfig(
        dim=args.dim,
        replicas=args.replicas,
        particles=args.chains,
        adapt_sweeps=args.adapt_sweeps,
        burnin_sweeps=args.burnin_sweeps,
        retained_sweeps=args.retained_per_chain,
        thin=args.thin,
        beta_hmc=args.hmc_beta,
        hmc_steps=args.hmc_steps,
        cold_hmc_probability=args.cold_hmc_probability,
    )
    if args.observation_mode == "fixed":
        full_observation = np.full(
            args.observed_blocks, args.remaining_observation, dtype=float
        )
        full_observation[: args.leading_blocks] = args.leading_observation
    else:
        full_observation = make_datasets(
            args.scenario,
            args.dim // 2,
            "prior-predictive",
            (args.observation_seed,),
            (),
            args.generated_noise_sigma,
            observation_tilt=args.observation_tilt,
            banana_tau=args.banana_tau, sine_tau=args.sine_tau,
        )[0][1]
    problem = CurvedObservationProblem(
        args.scenario,
        sampler_cfg,
        args.sigma_y,
        full_observation[: args.observed_blocks],
        observed_blocks=args.observed_blocks,
        observation_tilt=args.observation_tilt,
        banana_tau=args.banana_tau, sine_tau=args.sine_tau,
    )

    data_generator = torch.Generator(device=device).manual_seed(args.training_seed + 1)
    with torch.no_grad():
        training_data = problem.sample_data_torch(
            args.training_samples,
            device=device,
            dtype=dtype,
            generator=data_generator,
        )
    training_data_sha256 = hashlib.sha256(
        training_data.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    baseline_network_cfg = NetworkConfig(
        dim=args.dim,
        hidden=args.network_hidden,
        depth=args.network_depth,
        time_harmonics=args.time_harmonics,
    )
    network_configs = {
        kind: (NetworkConfig(dim=args.dim, hidden=args.network_hidden,
                             depth=args.imf_network_depth, time_harmonics=0)
               if kind == "imf" else baseline_network_cfg)
        for kind in model_kinds(methods)
    }
    training_cfg = TrainingConfig(
        steps=args.training_steps,
        batch_size=args.training_batch_size,
        learning_rate=args.learning_rate,
        fm_coupling=args.fm_coupling,
        imf_coupling=args.imf_coupling,
        imf_auxiliary_weight=args.imf_auxiliary_weight,
        imf_learning_rate=args.imf_learning_rate,
        imf_adam_beta1=args.imf_adam_beta1,
        imf_adam_beta2=args.imf_adam_beta2,
        grad_clip=args.grad_clip,
        imf_grad_clip=args.imf_grad_clip,
        imf_weight_decay=args.imf_weight_decay,
        imf_time_distribution=args.imf_time_distribution,
        imf_time_mean=args.imf_time_mean,
        imf_time_std=args.imf_time_std,
        imf_equal_time_fraction=args.imf_equal_time_fraction,
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    models: dict[str, torch.nn.Module] = {}
    training_rows: list[dict[str, object]] = []
    for kind in model_kinds(methods):
        network_cfg = network_configs[kind]
        implementation_suffix = (
            "" if MODEL_IMPLEMENTATION_VERSION[kind] == 1
            else f"_v{MODEL_IMPLEMENTATION_VERSION[kind]}"
        )
        checkpoint_path = args.checkpoint_dir / (
            f"{GEOMETRY_VERSION}_{args.scenario}_d{args.dim}_"
            f"h{network_cfg.hidden}_depth{network_cfg.depth}_n{args.training_samples}_seed{args.training_seed}_{kind}"
            f"{implementation_suffix}.pt"
        )
        if args.reuse_checkpoints and checkpoint_path.exists():
            loaded_kind, model, loaded_network, loaded_training, metadata = load_checkpoint(
                checkpoint_path, device, dtype
            )
            loaded_version = int(metadata.get("implementation_version", 1))
            training_differences = training_config_differences(
                loaded_training, training_cfg, kind
            )
            mismatch_details = []
            if metadata.get("training_data_sha256") != training_data_sha256:
                mismatch_details.append("training dataset differs or its fingerprint is missing")
            if metadata.get("prior_signature") != problem.prior_signature:
                mismatch_details.append("prior geometry/distribution differs; retrain this checkpoint")
            if loaded_kind != kind:
                mismatch_details.append(
                    f"model kind: saved={loaded_kind!r}, current={kind!r}"
                )
            if loaded_network != network_cfg:
                mismatch_details.append(
                    f"network config: saved={loaded_network!r}, current={network_cfg!r}"
                )
            mismatch_details.extend(
                f"training.{field}: saved={saved!r}, current={current!r}"
                for field, (saved, current) in training_differences.items()
            )
            if loaded_version != MODEL_IMPLEMENTATION_VERSION[kind]:
                mismatch_details.append(
                    "implementation version: "
                    f"saved={loaded_version}, "
                    f"current={MODEL_IMPLEMENTATION_VERSION[kind]}"
                )
            if mismatch_details:
                details = "\n  - ".join(mismatch_details)
                raise ValueError(
                    f"checkpoint configuration mismatch for {checkpoint_path}:\n"
                    f"  - {details}\n"
                    "Disable --reuse-checkpoints or use a separate checkpoint directory "
                    "if you intend to retrain this model family."
                )
            training_metadata = dict(metadata)
            training_metadata.setdefault("kind", kind)
            training_metadata.setdefault("parameter_count", parameter_count(model))
            training_metadata.setdefault("training_time_sec", float("nan"))
            training_metadata.setdefault("final_loss", float("nan"))
            training_metadata["loaded_checkpoint"] = True
        else:
            torch.manual_seed(args.training_seed + {"imf": 0, "fm": 1, "diffusion": 2}[kind])
            model = build_model(kind, network_cfg).to(device=device, dtype=dtype)
            training_metadata = train_learned_prior(
                kind,
                model,
                training_data,
                training_cfg,
                seed=args.training_seed + {"imf": 0, "fm": 1, "diffusion": 2}[kind],
            )
            training_metadata["loaded_checkpoint"] = False
            training_metadata["prior_signature"] = problem.prior_signature
            training_metadata["training_data_sha256"] = training_data_sha256
            training_metadata["training_samples"] = args.training_samples
            training_metadata["training_seed"] = args.training_seed
            save_checkpoint(
                checkpoint_path,
                kind,
                model,
                network_cfg,
                training_cfg,
                training_metadata,
            )
        models[kind] = model
        training_rows.append(training_metadata)

    transports: dict[str, LearnedTransport] = {}
    if "imf" in models:
        transports["imf"] = LearnedTransport(
            "imf", models["imf"], steps=args.imf_steps, solver="euler"
        )
    if "fm" in models:
        transports["fm"] = LearnedTransport(
            "fm", models["fm"], steps=args.fm_steps, solver=args.fm_solver,
            rtol=args.fm_rtol, atol=args.fm_atol,
        )

    # Prior diagnostics use fresh samples, independent of the fixed training
    # dataset, and are computed before any likelihood-guided sampling.
    prior_reference_generator = torch.Generator(device=device).manual_seed(
        args.training_seed + 2
    )
    with torch.no_grad():
        true_prior = problem.sample_data_torch(
            args.prior_diagnostic_samples,
            device=device,
            dtype=dtype,
            generator=prior_reference_generator,
        ).cpu().numpy()
    true_prior_active = problem.full_coordinates(true_prior)
    generated_priors_active: dict[str, np.ndarray] = {}
    prior_rows: list[dict[str, object]] = []
    for row in training_rows:
        kind = str(row["kind"])
        if kind in transports:
            generated = sample_learned_prior(
                transports[kind],
                args.prior_diagnostic_samples,
                args.dim,
                batch_size=args.dps_batch_size,
                seed=args.training_seed + 200 + len(kind),
            ).numpy()
        elif kind == "diffusion":
            generated = sample_diffusion_prior(
                models[kind],
                samples=args.prior_diagnostic_samples,
                dim=args.dim,
                training_cfg=training_cfg,
                batch_size=args.dps_batch_size,
                seed=args.training_seed + 200 + len(kind),
            ).numpy()
        else:
            continue
        generated_active = problem.full_coordinates(generated)
        generated_priors_active[kind] = generated_active
        from evaluation import sliced_w2
        prior_sliced_w2 = sliced_w2(
            generated_active,
            true_prior_active,
            args.sliced_directions,
            seed=args.training_seed + 303,
        )
        mean_rmse = float(np.sqrt(np.mean(
            (generated_active.mean(axis=0) - true_prior_active.mean(axis=0)) ** 2
        )))
        reference_covariance = np.cov(true_prior_active, rowvar=False)
        generated_covariance = np.cov(generated_active, rowvar=False)
        covariance_relative_frobenius = float(
            np.linalg.norm(generated_covariance - reference_covariance, ord="fro")
            / max(np.linalg.norm(reference_covariance, ord="fro"), 1e-12)
        )
        row["prior_full_data_sliced_w2"] = prior_sliced_w2
        row["prior_full_data_mean_rmse"] = mean_rmse
        row["prior_full_data_cov_relative_frobenius"] = covariance_relative_frobenius
        prior_rows.append({
            "model": kind,
            "diagnostic_samples": args.prior_diagnostic_samples,
            "full_data_sliced_w2": prior_sliced_w2,
            "full_data_mean_rmse": mean_rmse,
            "full_data_cov_relative_frobenius": covariance_relative_frobenius,
        })

    make_prior_diagnostic_plots(
        args.outdir,
        true_prior_active,
        generated_priors_active,
        max_points=args.plot_points,
        heatmap_bins=args.heatmap_bins,
    )

    pair_source, pair_data, bin_edges_source, bin_weights_source = [], [], [], []
    for block, observation in enumerate(problem.observation_values):
        source, data, edges, weights = make_pair_reference(
            problem,
            float(observation),
            args.reference_grid_n,
            args.reference_samples,
            seed=9101 + block,
        )
        pair_source.append(source)
        pair_data.append(data)
        bin_edges_source.append(edges)
        bin_weights_source.append(weights)
    del pair_source, bin_edges_source, bin_weights_source
    data_reference = make_active_reference(
        pair_data, args.reference_samples, seed=11_021
    )
    data_bin_edges, data_bin_weights = make_reference_bins(
        data_reference, args.observed_blocks
    )

    sampler_seeds = csv_tuple(args.sampler_seeds, int)
    metric_rows: list[dict[str, object]] = []
    plot_results: list[RunResult] = []
    plot_labels: list[str] = []
    initial_schedule = np.full(args.replicas, args.initial_hmc_epsilon)

    for seed in sampler_seeds:
        for method in methods:
            if method == "diffusion_dps":
                sample_count = args.retained_per_chain * args.chains
                samples, timing = dps_sample(
                    models["diffusion"],
                    problem,
                    samples=sample_count,
                    dim=args.dim,
                    training_cfg=training_cfg,
                    guidance_scale=args.dps_guidance_scale,
                    batch_size=args.dps_batch_size,
                    seed=seed + 50_000,
                )
                result = dps_result(
                    samples,
                    problem,
                    retained_per_chain=args.retained_per_chain,
                    chains=args.chains,
                    seed=seed,
                    timing=timing,
                    diffusion_steps=args.diffusion_steps,
                )
                model_name, sampler_name = "diffusion", "DPS"
                nfe = args.diffusion_steps
            else:
                kind = "imf" if method.startswith("imf_") else "fm"
                transport = transports[kind]
                calls_before = transport.map_calls
                nfe_before = transport.function_evaluations
                posterior = LearnedTransportPosterior(problem, transport)
                if method.endswith("spt_pcn"):
                    result = TorchSPT(
                        posterior, sampler_cfg, "pcn", seed,
                        target_hmc_acceptance=args.target_split_hmc_acceptance,
                    ).run()
                    sampler_name = "SPT+pCN"
                elif method.endswith("spt_hybrid"):
                    result = TorchSPT(
                        posterior,
                        sampler_cfg,
                        "full_split_hmc_mixture",
                        seed,
                        hmc_epsilon_schedule=initial_schedule,
                        target_hmc_acceptance=args.target_split_hmc_acceptance,
                    ).run()
                    sampler_name = "SPT+pCN/split-HMC"
                elif method == "fm_sgfm_hmc":
                    result = TorchClassicalHMC(
                        posterior,
                        sampler_cfg,
                        seed,
                        initial_epsilon=args.initial_hmc_epsilon,
                        target_acceptance=args.target_sgfm_hmc_acceptance,
                    ).run()
                    sampler_name = "classical HMC (no PT)"
                elif method == "fm_split_hmc":
                    no_tempering_cfg = SamplerConfig(
                        dim=args.dim,
                        replicas=1,
                        particles=args.chains,
                        adapt_sweeps=args.adapt_sweeps,
                        burnin_sweeps=args.burnin_sweeps,
                        retained_sweeps=args.retained_per_chain,
                        thin=args.thin,
                        beta_hmc=1.0,
                        hmc_steps=args.hmc_steps,
                        cold_hmc_probability=1.0,
                    )
                    result = TorchSPT(
                        posterior,
                        no_tempering_cfg,
                        "full_split_hmc_mixture",
                        seed,
                        hmc_epsilon_schedule=np.asarray([args.initial_hmc_epsilon]),
                        target_hmc_acceptance=args.target_split_hmc_acceptance,
                    ).run()
                    result.method = "fm_split_hmc"
                    sampler_name = "split HMC (no PT)"
                else:
                    raise AssertionError(method)
                result.method = method
                model_name = kind
                method_map_calls = transport.map_calls - calls_before
                method_nfe = transport.function_evaluations - nfe_before
                nfe = method_nfe / max(method_map_calls, 1)

            training_row = next(row for row in training_rows if row["kind"] == model_name)
            row = fixed_output_metrics(
                result,
                problem,
                data_reference,
                data_bin_edges,
                data_bin_weights,
                sliced_directions=args.sliced_directions,
                model_name=model_name,
                sampler_name=sampler_name,
                transport_nfe=nfe,
                parameter_count=int(training_row["parameter_count"]),
                training_time_sec=float(training_row["training_time_sec"]),
            )
            row.update({
                "scenario": args.scenario,
                "dim": args.dim,
                "observed_blocks": args.observed_blocks,
                "prior_blocks": problem.num_blocks,
                "sigma_y": args.sigma_y,
                "observation_tilt": args.observation_tilt,
                "observation_mode": args.observation_mode,
                "leading_observation": args.leading_observation,
                "leading_blocks": args.leading_blocks,
                "remaining_observation": args.remaining_observation,
                "observation_seed": args.observation_seed,
            })
            apply_sampling_reporting(row)
            metric_rows.append(row)
            if seed == sampler_seeds[0]:
                plot_results.append(result)
                plot_labels.append(method)
            print(
                f"{method:20s} seed={seed} retained="
                f"{row['nominal_retained_samples']} runtime={row['runtime_sec']:.3f}s "
                f"SW2={row['data_active_sliced_w2']:.5f}",
                flush=True,
            )

    # Model families expose different loss components.  evaluation.write_csv
    # derives its header from the first row, so write a rectangular union of
    # all component fields rather than dropping family-specific diagnostics.
    write_csv(args.outdir / "training.csv", rectangularize_rows(training_rows))
    write_csv(args.outdir / "prior_diagnostics.csv", prior_rows)
    write_csv(args.outdir / "by_seed.csv", metric_rows)
    # A compact numeric summary avoids obscuring training-seed versus sampler-seed variation.
    summary_rows = []
    for method in methods:
        subset = [row for row in metric_rows if row["method"] == method]
        summary: dict[str, object] = {
            "method": method,
            "sampler_seeds": len(subset),
        }
        numeric_keys = [
            key for key, value in subset[0].items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and key not in {"seed"}
        ]
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in subset])
            finite = values[np.isfinite(values)]
            summary[f"{key}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            summary[f"{key}_std"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1
                else (0.0 if len(finite) == 1 else float("nan"))
            )
        summary_rows.append(summary)
    write_csv(args.outdir / "summary.csv", summary_rows)
    if args.leading_blocks > 0:
        make_posterior_heatmap(
            args.outdir / "posterior_leading_observation_heatmap.png",
            plot_results, plot_labels, data_reference, problem,
            block_index=0, seed=sampler_seeds[0], bins=args.heatmap_bins,
        )
        make_end_to_end_sample_plot(
            args.outdir / "posterior_leading_observation.png",
            plot_results,
            plot_labels,
            data_reference,
            problem,
            block_index=0,
            max_points=args.plot_points,
        )

    first_remaining_block = args.leading_blocks
    if first_remaining_block < args.observed_blocks:
        make_posterior_heatmap(
            args.outdir / "posterior_remaining_observation_heatmap.png",
            plot_results, plot_labels, data_reference, problem,
            block_index=first_remaining_block, seed=sampler_seeds[0], bins=args.heatmap_bins,
        )
        make_end_to_end_sample_plot(
            args.outdir / "posterior_remaining_observation.png",
            plot_results,
            plot_labels,
            data_reference,
            problem,
            block_index=first_remaining_block,
            max_points=args.plot_points,
        )

    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_end_to_end_report(
        args.outdir / "REPORT.md",
        arguments=arguments,
        training_rows=training_rows,
        metric_rows=metric_rows,
    )
    with (args.outdir / "REPORT.md").open("a") as report:
        report.write(
            "\n\nDPS reporting: ESS, ESS/second, R-hat, and acceptance "
            "diagnostics are not applicable and are recorded as NaN. DPS "
            "generates independent trajectories; independence does not imply "
            "posterior accuracy. The CSV field samples_per_sec reports nominal "
            "output count divided by sampling runtime for every method.\n"
        )
    (args.outdir / "protocol.json").write_text(json.dumps({
        "arguments": arguments,
        "methods": methods,
        "fixed_output_rule": {
            "retained_per_chain": args.retained_per_chain,
            "output_chains": args.chains,
            "nominal_samples_per_method": args.retained_per_chain * args.chains,
        },
        "network_fairness": {
            "shared_hidden": args.network_hidden,
            "imf_depth": args.imf_network_depth,
            "fm_diffusion_depth": args.network_depth,
            "imf_time_features": "raw r,t",
            "fm_diffusion_time_harmonics": args.time_harmonics,
            "identical_architectures": False,
            "shared_training_samples": args.training_samples,
            "shared_optimizer_updates": args.training_steps,
            "shared_batch_size": args.training_batch_size,
            "parameter_counts_reported_separately": True,
        },
        "imf_training_objective": {
            "data_noise_coupling": args.imf_coupling,
            "time_distribution": args.imf_time_distribution,
            "default_time_law": "independent Uniform[0,1] r,t; includes r>t",
            "logit_normal_mean": args.imf_time_mean,
            "logit_normal_std": args.imf_time_std,
            "equal_time_fraction": args.imf_equal_time_fraction,
            "loss": "meanflow_mse + lambda * auxiliary_velocity_mse (coordinate averages)",
            "auxiliary_velocity_definition": "v_theta(w,t) = head_v(backbone(w,t,t))",
            "auxiliary_weight": args.imf_auxiliary_weight,
            "adaptive_loss_weighting": False,
            "optimizer": "Adam",
            "weight_decay": args.imf_weight_decay,
            "gradient_clip": args.imf_grad_clip,
            "batch_sampling": "shuffled epochs including final short batch",
            "map": "z_next = z - (t-r)*head_u(backbone(z,r,t))",
            "map_steps": args.imf_steps,
            "implementation_version": MODEL_IMPLEMENTATION_VERSION["imf"],
            "theory_uses_independent_coupling": True,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
