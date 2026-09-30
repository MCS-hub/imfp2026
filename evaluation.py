"""Metrics, aggregation, plots, and reports for the SPT benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from curved_problem import CurvedObservationProblem
from spt_score import RunResult, SamplerConfig


def chain_ess(trace: np.ndarray) -> float:
    """Sum initial-positive-sequence ESS over independent particles."""
    t, particles = trace.shape
    total = 0.0
    for particle in range(particles):
        x = np.array(trace[:, particle], dtype=float, copy=True)
        x -= np.mean(x)
        variance = np.dot(x, x) / t
        if not np.isfinite(variance) or variance <= 1e-15:
            continue
        nfft = 1 << (2 * t - 1).bit_length()
        spectrum = np.fft.rfft(x, n=nfft)
        autocovariance = np.fft.irfft(
            spectrum * np.conjugate(spectrum), n=nfft
        )[:t]
        autocovariance /= np.arange(t, 0, -1)
        autocorrelation = autocovariance / autocovariance[0]
        tau = 1.0
        for lag in range(1, t - 1, 2):
            pair = autocorrelation[lag] + autocorrelation[lag + 1]
            if pair <= 0.0:
                break
            tau += 2.0 * pair
        total += min(t, t / max(tau, 1.0))
    return float(total)


def sliced_w2(
    sample: np.ndarray,
    reference: np.ndarray,
    directions: int,
    seed: int = 551,
) -> float:
    rng = np.random.default_rng(seed)
    projection_directions = rng.normal(size=(directions, sample.shape[1]))
    projection_directions /= np.linalg.norm(
        projection_directions, axis=1, keepdims=True
    )
    probabilities = (np.arange(2000) + 0.5) / 2000
    squared_distances = []
    for direction in projection_directions:
        sample_quantiles = np.quantile(sample @ direction, probabilities)
        reference_quantiles = np.quantile(reference @ direction, probabilities)
        squared_distances.append(np.mean((sample_quantiles - reference_quantiles) ** 2))
    return float(np.sqrt(np.mean(squared_distances)))


def coordinate_ess(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace)
    flattened = trace.reshape(trace.shape[0], trace.shape[1], -1)
    return np.asarray([chain_ess(flattened[..., j]) for j in range(flattened.shape[-1])])


def split_rhat(trace: np.ndarray) -> np.ndarray:
    """Coordinatewise split-Rhat, treating retained particles as chains."""
    trace = np.asarray(trace)
    flattened = trace.reshape(trace.shape[0], trace.shape[1], -1)
    half = flattened.shape[0] // 2
    if half < 4 or flattened.shape[1] < 2:
        return np.full(flattened.shape[-1], np.nan)
    split = np.concatenate((flattened[:half], flattened[-half:]), axis=1)
    chain_means = np.mean(split, axis=0)
    within = np.mean(np.var(split, axis=0, ddof=1), axis=0)
    between = half * np.var(chain_means, axis=0, ddof=1)
    variance = ((half - 1.0) / half) * within + between / half
    return np.sqrt(np.divide(variance, within, out=np.full_like(variance, np.nan), where=within > 0))


def source_metrics(
    result,
    problem: CurvedObservationProblem,
    active_reference: np.ndarray,
    bin_edges: list[np.ndarray],
    bin_weights: list[np.ndarray],
    cfg: SamplerConfig,
    gradient_cost: int,
    sliced_directions: int,
) -> dict[str, object]:
    trace = result.active_trace
    active = trace.reshape(-1, problem.active_dim)
    pairs = active.reshape(-1, problem.observed_blocks, 2)
    tv = []
    for block in range(problem.observed_blocks):
        labels = np.clip(np.digitize(pairs[:, block, 0], bin_edges[block][1:-1]), 0, 7)
        observed = np.bincount(labels, minlength=8) / labels.size
        tv.append(0.5 * np.sum(np.abs(observed - bin_weights[block])))
    block_trace = trace.reshape(trace.shape[0], trace.shape[1], problem.observed_blocks, 2)
    all_ess = coordinate_ess(block_trace)
    u_ess = coordinate_ess(block_trace[..., 0])
    v_ess = coordinate_ess(block_trace[..., 1])
    data_blocks = problem.transport_blocks(block_trace)
    # Compare both measures in the observed data coordinates. The incoming
    # reference is in source space, so it must pass through the exact map too.
    data_active = data_blocks.reshape(-1, problem.active_dim)
    data_reference = problem.transport_blocks(
        np.asarray(active_reference).reshape(-1, problem.observed_blocks, 2)
    ).reshape(-1, problem.active_dim)
    finite_data_rows = np.all(np.isfinite(data_active), axis=1)
    data_sw2 = (
        sliced_w2(
            data_active[finite_data_rows], data_reference,
            int(sliced_directions), result.seed + 813,
        )
        if np.count_nonzero(finite_data_rows) >= 2
        and np.all(np.isfinite(data_reference))
        else float("nan")
    )
    residual_trace = (
        problem.observe_blocks(data_blocks) - problem.observation_values
    )
    residual_ess = coordinate_ess(residual_trace)
    phi_trace, _ = problem.phi_active_grad(trace)
    phi_ess = float(chain_ess(phi_trace))
    rhat = split_rhat(block_trace)
    calls = result.phi_evaluations + gradient_cost * result.gradient_evaluations
    scale = 1000.0 / calls
    return {
        "method": result.method,
        "seed": result.seed,
        "data_active_sliced_w2": data_sw2,
        "data_finite_sample_fraction": float(np.mean(finite_data_rows)),
        "active_sliced_w2": sliced_w2(
            active,
            active_reference,
            min(48, int(sliced_directions)),
            result.seed + 47,
        ),
        "mean_block_bin_tv": float(np.mean(tv)),
        "active_ess_mean_per_1k_calls": float(np.mean(all_ess) * scale),
        "active_ess_median_per_1k_calls": float(np.median(all_ess) * scale),
        "active_ess_min_per_1k_calls": float(np.min(all_ess) * scale),
        "u_ess_mean_per_1k_calls": float(np.mean(u_ess) * scale),
        "u_ess_min_per_1k_calls": float(np.min(u_ess) * scale),
        "v_ess_mean_per_1k_calls": float(np.mean(v_ess) * scale),
        "v_ess_min_per_1k_calls": float(np.min(v_ess) * scale),
        "residual_ess_mean_per_1k_calls": float(np.mean(residual_ess) * scale),
        "residual_ess_min_per_1k_calls": float(np.min(residual_ess) * scale),
        "log_likelihood_ess_per_1k_calls": float(phi_ess * scale),
        # Backward-compatible name: now the mean u-coordinate ESS.
        "slow_ess_per_1k_calls": float(np.mean(u_ess) * scale),
        "active_split_rhat_median": float(np.nanmedian(rhat)) if np.any(np.isfinite(rhat)) else float("nan"),
        "active_split_rhat_max": float(np.nanmax(rhat)) if np.any(np.isfinite(rhat)) else float("nan"),
        "cold_acceptance": float(result.local_acceptance[-1]),
        "cold_hmc_acceptance": float(result.hmc_acceptance[-1]),
        "cold_pcn_acceptance": float(result.pcn_acceptance[-1]),
        "cold_proposal_scale": float(result.proposal_scale[-1]),
        "hmc_fraction": float(result.hmc_move_fraction[-1]),
        "min_swap_acceptance": float(np.min(result.swap_acceptance)),
        "runtime_sec": result.runtime_sec,
        "phi_evaluations": int(result.phi_evaluations),
        "gradient_evaluations": int(result.gradient_evaluations),
        "gradient_cost": int(gradient_cost),
        "total_oracle_calls": calls,
    }


def metrics(result, problem, *args, **kwargs):
    """Retain source diagnostics and evaluate the transported scalar sequences.

    Only observed blocks enter either coordinate aggregate. Preserve the
    (retained sweep, chain, block, coordinate) axes for autocorrelation estimation.
    The exact map evaluation here is postprocessing, outside sampling cost.
    """
    row = source_metrics(result, problem, *args, **kwargs)
    source = np.asarray(result.active_trace)
    blocks = source.reshape(source.shape[:2] + (problem.observed_blocks, 2))
    data = problem.transport_blocks(blocks)
    if np.all(np.isfinite(data)):
        ess = coordinate_ess(data).reshape(problem.observed_blocks, 2)
    else:
        ess = np.full((problem.observed_blocks, 2), np.nan)
    calls = float(row["total_oracle_calls"])
    scale = 1000.0 / calls if calls > 0 else float("nan")
    for name, reducer in (("mean", np.mean), ("median", np.median), ("min", np.min)):
        value = float(reducer(ess))
        row[f"data_coordinate_ess_{name}"] = value
        row[f"data_coordinate_ess_{name}_per_1k_calls"] = value * scale
    row["data_x1_ess_mean_per_1k_calls"] = float(np.mean(ess[:, 0])) * scale
    row["data_x2_ess_mean_per_1k_calls"] = float(np.mean(ess[:, 1])) * scale
    return row

METRICS = (
    "active_sliced_w2",
    "data_active_sliced_w2",
    "data_finite_sample_fraction",
    "mean_block_bin_tv",
    "active_ess_mean_per_1k_calls",
    "active_ess_median_per_1k_calls",
    "active_ess_min_per_1k_calls",
    "u_ess_mean_per_1k_calls",
    "u_ess_min_per_1k_calls",
    "v_ess_mean_per_1k_calls",
    "v_ess_min_per_1k_calls",
    "residual_ess_mean_per_1k_calls",
    "residual_ess_min_per_1k_calls",
    "log_likelihood_ess_per_1k_calls",
    "slow_ess_per_1k_calls",
    "active_split_rhat_median",
    "active_split_rhat_max",
    "cold_acceptance",
    "cold_hmc_acceptance",
    "cold_pcn_acceptance",
    "cold_proposal_scale",
    "hmc_fraction",
    "min_swap_acceptance",
    "runtime_sec",
    "phi_evaluations",
    "gradient_evaluations",
    "total_oracle_calls",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted({
        (
            str(row["scenario"]), str(row["dataset_id"]),
            int(row["observed_blocks"]), float(row["sigma_y"]), str(row["method"]),
        )
        for row in rows
    })
    for scenario, dataset_id, blocks, sigma, method in keys:
        subset = [
            row for row in rows
            if row["scenario"] == scenario
            and row["dataset_id"] == dataset_id
            and row["observed_blocks"] == blocks
            and row["sigma_y"] == sigma
            and row["method"] == method
        ]
        record: dict[str, object] = {
            "scenario": scenario,
            "dataset_id": dataset_id,
            "observed_blocks": blocks, "dim": int(subset[0]["dim"]),
            "prior_blocks": int(subset[0]["dim"]) // 2,
            "active_dimension": 2 * blocks,
            "sigma_y": sigma,
            "observation_mean": float(subset[0]["observation_mean"]),
            "observation_std": float(subset[0]["observation_std"]),
            "method": method,
            "seeds": len(subset),
            "total_gn_trace": float(subset[0]["total_gn_trace"]),
            "gn_effective_rank": float(subset[0]["gn_effective_rank"]),
            "gn_lambda95": float(subset[0]["gn_lambda95"]),
            "initial_cold_epsilon": float(subset[0]["initial_cold_epsilon"]),
            "hmc_steps": int(subset[0]["hmc_steps"]),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in subset])
            finite = values[np.isfinite(values)]
            record[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            record[f"{metric}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        output.append(record)
    return output


DATA_ESS_METRICS = (
    "data_coordinate_ess_mean", "data_coordinate_ess_median",
    "data_coordinate_ess_min", "data_coordinate_ess_mean_per_1k_calls",
    "data_coordinate_ess_median_per_1k_calls",
    "data_coordinate_ess_min_per_1k_calls",
    "data_x1_ess_mean_per_1k_calls", "data_x2_ess_mean_per_1k_calls",
)

def summarize(rows):
    summary = source_summarize(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(condition_key(row) + (row["method"],), []).append(row)
    for record in summary:
        subset = grouped[condition_key(record) + (record["method"],)]
        for metric in DATA_ESS_METRICS:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            finite = values[np.isfinite(values)]
            record[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            record[f"{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1
                else (0.0 if finite.size == 1 else float("nan"))
            )
    return summary

def safe_ratio(numerator: float, denominator: float) -> float:
    """A zero/invalid baseline makes a performance ratio undefined."""
    if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(numerator):
        return float("nan")
    return float(numerator / denominator)


def source_crossover(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    conditions = sorted({
        (str(row["scenario"]), str(row["dataset_id"]), int(row["observed_blocks"]), float(row["sigma_y"]))
        for row in summary
    })
    for scenario, dataset_id, blocks, sigma in conditions:
        selected = [
            row for row in summary
            if row["scenario"] == scenario
            and row["dataset_id"] == dataset_id
            and row["observed_blocks"] == blocks
            and row["sigma_y"] == sigma
        ]
        by_method = {str(row["method"]): row for row in selected}
        pcn = by_method["pcn_equal_calls"]
        hybrid = by_method["full_split_hmc_mixture"]
        output.append({
            "scenario": scenario,
            "dataset_id": dataset_id,
            "observed_blocks": blocks, "dim": int(hybrid["dim"]),
            "prior_blocks": int(hybrid["dim"]) // 2,
            "active_dimension": 2 * blocks,
            "sigma_y": sigma,
            "per_direction_precision": 1.0 / sigma**2,
            "total_gn_trace": hybrid["total_gn_trace"],
            "gn_effective_rank": hybrid["gn_effective_rank"],
            "sw2_ratio_hybrid_over_pcn": safe_ratio(float(hybrid["active_sliced_w2_mean"]), float(pcn["active_sliced_w2_mean"])),
            "data_sw2_ratio_hybrid_over_pcn": safe_ratio(float(hybrid["data_active_sliced_w2_mean"]), float(pcn["data_active_sliced_w2_mean"])),
            "bin_tv_ratio_hybrid_over_pcn": safe_ratio(float(hybrid["mean_block_bin_tv_mean"]), float(pcn["mean_block_bin_tv_mean"])),
            "active_ess_min_ratio_hybrid_over_pcn": safe_ratio(float(hybrid["active_ess_min_per_1k_calls_mean"]), float(pcn["active_ess_min_per_1k_calls_mean"])),
            "active_ess_mean_ratio_hybrid_over_pcn": safe_ratio(float(hybrid["active_ess_mean_per_1k_calls_mean"]), float(pcn["active_ess_mean_per_1k_calls_mean"])),
            "slow_ess_ratio_hybrid_over_pcn": safe_ratio(float(hybrid["u_ess_mean_per_1k_calls_mean"]), float(pcn["u_ess_mean_per_1k_calls_mean"])),
            "residual_ess_ratio_hybrid_over_pcn": safe_ratio(float(hybrid["residual_ess_min_per_1k_calls_mean"]), float(pcn["residual_ess_min_per_1k_calls_mean"])),
            "hybrid_hmc_acceptance": hybrid["cold_hmc_acceptance_mean"],
            "hybrid_pcn_acceptance": hybrid["cold_pcn_acceptance_mean"],
            "hybrid_min_swap_acceptance": hybrid["min_swap_acceptance_mean"],
            "hybrid_initial_cold_epsilon": hybrid["initial_cold_epsilon"],
            "hybrid_tuned_cold_epsilon": hybrid["cold_proposal_scale_mean"],
            "hybrid_hmc_steps": hybrid["hmc_steps"],
        })
    return output


CONDITION_FIELDS = ("scenario", "dataset_id", "observed_blocks", "sigma_y")

def condition_key(row):
    return tuple(row[field] for field in CONDITION_FIELDS)

def crossover(summary):
    output = source_crossover(summary)
    lookup = {condition_key(row) + (row["method"],): row for row in summary}
    for row in output:
        pcn = lookup[condition_key(row) + ("pcn_equal_calls",)]
        hybrid = lookup[condition_key(row) + ("full_split_hmc_mixture",)]
        for statistic in ("mean", "median", "min"):
            metric = f"data_coordinate_ess_{statistic}_per_1k_calls_mean"
            numerator, denominator = float(hybrid[metric]), float(pcn[metric])
            row[f"data_ess_{statistic}_ratio_hybrid_over_pcn"] = (
                numerator / denominator
                if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0
                else float("nan")
            )
    return output

def make_heatmaps(
    path: Path,
    crossover_rows: list[dict[str, object]],
    block_values: tuple[int, ...],
    sigma_values: tuple[float, ...],
    scenarios: tuple[str, ...],
    *, space: str = "source",
) -> None:
    if space not in ("source", "data"):
        raise ValueError("space must be source or data")
    sw2_key = "data_sw2_ratio_hybrid_over_pcn" if space == "data" else "sw2_ratio_hybrid_over_pcn"
    ess_key = "data_ess_mean_ratio_hybrid_over_pcn" if space == "data" else "active_ess_mean_ratio_hybrid_over_pcn"
    metrics_to_plot = (
        (sw2_key, f"{space.capitalize()} SW2 ratio", "coolwarm_r", 1.0),
        (
            ess_key,
            f"{space.capitalize()} mean coordinate ESS ratio",
            "coolwarm",
            1.0,
        ),
    )
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(10, 4.2 * len(scenarios)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_index, scenario in enumerate(scenarios):
        for column_index, (metric, title, cmap, neutral) in enumerate(metrics_to_plot):
            matrix = np.empty((len(sigma_values), len(block_values)))
            for i, sigma in enumerate(sigma_values):
                for j, blocks in enumerate(block_values):
                    matches = [
                        row for row in crossover_rows
                        if row["scenario"] == scenario
                        and row["observed_blocks"] == blocks
                        and row["sigma_y"] == sigma
                    ]
                    matrix[i, j] = float(np.mean([float(match[metric]) for match in matches]))
            axis = axes[row_index, column_index]
            finite_deviation = np.abs(matrix[np.isfinite(matrix)] - neutral)
            maximum_deviation = max(float(np.max(finite_deviation)), 0.05) if finite_deviation.size else 0.05
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=neutral - maximum_deviation,
                vmax=neutral + maximum_deviation,
            )
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
            axis.set_xticks(range(len(block_values)), block_values)
            axis.set_yticks(range(len(sigma_values)), sigma_values)
            axis.set_xlabel("observed block count m")
            axis.set_ylabel("per-direction noise sigma_y")
            axis.set_title(f"{scenario}: {title} (hybrid / pCN)")
            fig.colorbar(image, ax=axis, shrink=0.82)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def make_data_ess_heatmaps(outdir, summary, ratios, block_values, sigma_values, scenarios):
    """Two figures; each uses one shared color scale across all its panels.

    Absolute values: seed means followed by equal-weight dataset means.
    Ratios: hybrid/pCN ratios of seed means, followed by dataset means,
    matching the existing source-space crossover convention.
    """
    specifications = (
        ("data_ess_heatmaps.png", summary,
         (("pcn_equal_calls", "SPT + pCN", "data_coordinate_ess_mean_per_1k_calls_mean"),
          ("full_split_hmc_mixture", "SPT + hybrid", "data_coordinate_ess_mean_per_1k_calls_mean")),
         "Data-space mean coordinate ESS per 1,000 oracle calls", False),
        ("data_ess_ratio_heatmaps.png", ratios,
         ((None, "Mean coordinate ESS", "data_ess_mean_ratio_hybrid_over_pcn"),
          (None, "Minimum coordinate ESS", "data_ess_min_ratio_hybrid_over_pcn")),
         "Data-space ESS efficiency: hybrid / pCN (greater than 1 favors hybrid)", True),
    )
    _draw_data_metric_heatmaps(outdir, specifications, block_values, sigma_values, scenarios)


def make_data_sw2_heatmaps(outdir, summary, block_values, sigma_values, scenarios):
    """Absolute data SW2: seed means, then equal-weight means across datasets.

    Compare both methods on one shared linear scale across all scenarios.
    Only the observed 2m data coordinates are included; lower SW2 is better.
    """
    specifications = (
        ("data_sw2_heatmaps.png", summary,
         (("pcn_equal_calls", "SPT + pCN", "data_active_sliced_w2_mean"),
          ("full_split_hmc_mixture", "SPT + hybrid", "data_active_sliced_w2_mean")),
         "Data-space sliced W2 (lower is better)", False),
    )
    _draw_data_metric_heatmaps(outdir, specifications, block_values, sigma_values, scenarios)


def _draw_data_metric_heatmaps(outdir, specifications, block_values, sigma_values, scenarios):
    """Render metric matrices with one shared normalization per figure."""
    for filename, records, columns, title, is_ratio in specifications:
        is_sw2 = columns[0][2] == "data_active_sliced_w2_mean"
        matrices = []
        for scenario in scenarios:
            for method, _, metric in columns:
                matrix = np.full((len(sigma_values), len(block_values)), np.nan)
                for i, sigma in enumerate(sigma_values):
                    for j, blocks in enumerate(block_values):
                        values = np.asarray([
                            row[metric] for row in records
                            if row["scenario"] == scenario and row["sigma_y"] == sigma
                            and row["observed_blocks"] == blocks
                            and (method is None or row["method"] == method)
                        ], dtype=float)
                        # A missing/undefined dataset value must not be silently dropped.
                        if values.size and np.all(np.isfinite(values)):
                            matrix[i, j] = float(np.mean(values))
                matrices.append(matrix)
        finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
        if is_ratio:
            lower = min(0.95, float(np.min(finite))) if finite.size else 0.95
            upper = max(1.05, float(np.max(finite))) if finite.size else 1.05
            norm = TwoSlopeNorm(vmin=lower, vcenter=1.0, vmax=upper)
            cmap = plt.get_cmap("coolwarm").copy()
        else:
            norm = Normalize(vmin=0.0, vmax=max(float(np.max(finite)), 1e-12) if finite.size else 1.0)
            cmap = plt.get_cmap("viridis_r" if is_sw2 else "viridis").copy()
        cmap.set_bad("#e5e7eb")
        fig, axes = plt.subplots(len(scenarios), 2, squeeze=False,
                                 figsize=(11, 3.8 * len(scenarios) + 0.7), layout="constrained")
        for index, matrix in enumerate(matrices):
            axis = axes.flat[index]
            image = axis.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="auto",
                                cmap=cmap, norm=norm)
            for i, j in np.ndindex(matrix.shape):
                value = matrix[i, j]
                color = "black"
                if np.isfinite(value):
                    red, green, blue, _ = cmap(norm(value))
                    color = "white" if 0.2126*red + 0.7152*green + 0.0722*blue < 0.5 else "black"
                axis.text(j, i, (f"{value:.3g}" if is_sw2 else f"{value:.2f}") if np.isfinite(value) else "N/A",
                          ha="center", va="center", color=color)
            axis.set_xticks(range(len(block_values)), block_values)
            axis.set_yticks(range(len(sigma_values)), [f"{s:g}" for s in sigma_values])
            axis.set_xlabel("Observed block count m (fixed full dimension)")
            axis.set_ylabel(r"Noise $\sigma_y$")
            axis.set_title(f"{scenarios[index // 2]}: {columns[index % 2][1]}")
        fig.suptitle(title + "\nObserved blocks only", fontsize=12)
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85,
                     label="Hybrid / pCN" if is_ratio else ("Data-space sliced W2" if is_sw2 else "ESS per 1,000 oracle calls"))
        fig.savefig(outdir / filename, dpi=190)
        plt.close(fig)

def make_sample_comparison(
    path: Path,
    pcn_result: RunResult,
    hybrid_result: RunResult,
    active_reference: np.ndarray,
    observation: np.ndarray,
    scenario: str,
    dataset_id: str,
    sigma_y: float,
    max_points: int,
    observation_tilt: float = 0.0,
    *, block_index: int = 0, space: str = "source",
    problem: CurvedObservationProblem | None = None,
) -> None:
    """Plot a selected full-coordinate pair in source or data space."""
    blocks = len(observation)
    block = int(block_index)
    if not 0 <= block < blocks or space not in {"source", "data"}:
        raise ValueError("invalid block index or coordinate space")
    reference = active_reference.reshape(-1, blocks, 2)[:, block]
    pcn = pcn_result.active_trace.reshape(-1, blocks, 2)[:, block]
    hybrid = hybrid_result.active_trace.reshape(-1, blocks, 2)[:, block]

    if space == "data":
        if problem is None:
            raise ValueError("data-space plotting requires the problem")
        reference, pcn, hybrid = [problem.transport_blocks(x) for x in (reference, pcn, hybrid)]
    point_count = min(int(max_points), len(reference), len(pcn), len(hybrid))
    if point_count < 1:
        return
    rng = np.random.default_rng(20260831 + block)

    def subsample(values: np.ndarray) -> np.ndarray:
        if len(values) == point_count:
            return values
        return values[rng.choice(len(values), size=point_count, replace=False)]

    shown = [subsample(values) for values in (reference, pcn, hybrid)]
    combined = reference
    lower = np.quantile(combined, 0.0025, axis=0)
    upper = np.quantile(combined, 0.9975, axis=0)
    padding = 0.06 * np.maximum(upper - lower, 1e-6)
    lower, upper = lower - padding, upper + padding

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    titles = ("Reference", "SPT+pCN", "SPT+hybrid")
    colors = ("0.25", "tab:blue", "tab:orange")
    for axis, values, title, color in zip(axes, shown, titles, colors):
        axis.scatter(
            values[:, 0], values[:, 1], s=6, alpha=0.22,
            color=color, linewidths=0, rasterized=True,
        )
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_xlabel(fr"$u_{{{block + 1}}}$")
        axis.set_ylabel(fr"$v_{{{block + 1}}}$")
        inside = np.all((values >= lower) & (values <= upper), axis=1)
        axis.set_title(f"{title}; in window={100 * np.mean(inside):.1f}%")
        if space == "data":
            axis.set_xlabel(fr"data $x_{{{2 * block + 1}}}$")
            axis.set_ylabel(fr"data $x_{{{2 * block + 2}}}$")
            line = np.linspace(lower[0], upper[0], 100)
            axis.plot(line, observation[block] - observation_tilt * line, "k--", lw=1)
        axis.grid(alpha=0.18)
    fig.suptitle(
        f"{scenario}; {dataset_id}; m={blocks}; sigma_y={sigma_y:g}; "
        f"G(x)=x2+{observation_tilt:g}*x1; "
        f"block {block + 1} (y={observation[block]:.3g}); n={point_count} per panel"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def write_report(
    path: Path,
    args: argparse.Namespace,
    summary: list[dict[str, object]],
    crossover_rows: list[dict[str, object]],
    validation: dict[str, object],
) -> None:
    pcn_rows = [row for row in summary if row["method"] == "pcn_equal_calls"]
    max_pcn_target_gap = max(
        (abs(float(row["cold_pcn_acceptance_mean"]) - 0.30) for row in pcn_rows),
        default=float("nan"),
    )
    log_rank = np.log([float(row["observed_blocks"]) for row in crossover_rows])
    log_precision = np.log([float(row["per_direction_precision"]) for row in crossover_rows])
    log_sw2_ratio = np.log([float(row["sw2_ratio_hybrid_over_pcn"]) for row in crossover_rows])
    log_ess_ratio = np.log([
        float(row["active_ess_mean_ratio_hybrid_over_pcn"])
        for row in crossover_rows
    ])
    design = np.column_stack((np.ones(len(crossover_rows)), log_rank, log_precision))

    def factorial_fit(y: np.ndarray) -> tuple[np.ndarray, float]:
        valid = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
        y, x = y[valid], design[valid]
        if len(y) < x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
            return np.full(design.shape[1], np.nan), float("nan")
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        denominator = np.sum((y - np.mean(y)) ** 2)
        r2 = float("nan") if denominator <= 0 else float(
            1.0 - np.sum((y - x @ coefficients) ** 2) / denominator
        )
        return coefficients, r2

    sw2_coef, sw2_r2 = factorial_fit(log_sw2_ratio)
    ess_coef, ess_r2 = factorial_fit(log_ess_ratio)
    lines = [
        "# Fixed-prior observed-rank--precision crossover experiment",
        "",
        "## Design",
        "",
        f"- Fixed full dimension d={args.dim}, with {args.dim // 2} curved prior blocks. Separate factors are observed block count m and inverse noise variance 1/sigma_y^2.",
        f"- Observation mode: {args.observation_mode}; multiple datasets are kept separate before aggregation.",
        f"- Data-space observation: G_c(x1,x2)=x2+c*x1 with c={args.observation_tilt:g}.",
        "- Controlled mode uses a fixed number of leading-observation blocks and adds remaining-observation blocks. Modality depends on scenario and noise; the sine remainder is not assumed unimodal.",
        "- Baseline: SPT+pCN at every temperature.",
        f"- Hybrid: hot pCN and a cold split-HMC/pCN mixture with nominal HMC probability {args.cold_hmc_probability:g}.",
        f"- HMC is used for beta >= {args.hmc_beta:g}; all proposals and swaps are Metropolis corrected.",
        "- Every condition has a grid-accurate posterior reference and a common oracle-call budget.",
        f"- One joint value-and-gradient evaluation is charged as {args.gradient_cost} value-only oracle calls.",
        "- Both arms call the same pCN adaptation function. An optional zero-HMC mixture control checks this path directly.",
        "- HMC step sizes start from a prior-pilot curvature rule, adapt only during warmup, and are then frozen.",
        f"- Warmup targets HMC acceptance {args.target_hmc_acceptance:g}; inspect the separately reported HMC acceptance rather than aggregate mixture acceptance.",
        "- When enabled, the number of HMC steps increases in stiff conditions to preserve the target trajectory angle; all added gradients are charged to the oracle budget.",
        "",
        "## Two-factor crossover regression",
        "",
        f"For log(hybrid/pCN SW2): observed-block coefficient={sw2_coef[1]:.3f}, precision coefficient={sw2_coef[2]:.3f}, R^2={sw2_r2:.3f}.",
        f"For log(hybrid/pCN mean-coordinate ESS): observed-block coefficient={ess_coef[1]:.3f}, precision coefficient={ess_coef[2]:.3f}, R^2={ess_r2:.3f}.",
        "Rank and precision are not collapsed into m/sigma_y^2: the former changes the observed prefix at fixed dimension and fixed prior, while the latter changes their stiffness.",
        "",
        "## Diagnostics to inspect",
        "",
        "- `crossover.csv` contains the hybrid/pCN ratios for every factorial condition.",
        "- `crossover_heatmaps.png` visualizes whether the advantage grows with observed block count and per-direction precision.",
        "- Metrics use the observed-coordinate marginal, excluding prior-only blocks from the aggregate errors and ESS.",
        "- `sample_plots/` compares a common active-coordinate block under the reference, equal-cost pCN, and the hybrid; each panel uses the same number of samples and common axes.",
        "- Mean, median, and worst-coordinate ESS are reported over both u and v, together with residual and log-likelihood ESS and split-Rhat.",
        "- HMC and pCN acceptances are reported separately; their aggregate acceptance is not used for tuning.",
        "- Minimum swap acceptance must remain healthy, otherwise temperature-ladder failure confounds the result.",
        f"- Largest absolute deviation of the baseline cold pCN acceptance from its 0.30 adaptation target: {max_pcn_target_gap:.3f}.",
        "  A large value indicates incomplete pCN tuning and should trigger a longer adaptation or pilot initialization before interpreting the crossover.",
        "- This factorized benchmark isolates scaling effects; a dense nonseparable benchmark is still required for a general geometry claim.",
        "",
        "## Validation",
        "",
        f"```json\n{json.dumps(validation, indent=2)}\n```",
        "",
    ]
    path.write_text("\n".join(lines))


def make_reference_bins(
    data_active_reference: np.ndarray,
    observed_blocks: int,
    bins: int = 8,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Construct common data-space bins and reference probabilities."""
    pairs = np.asarray(data_active_reference).reshape(-1, observed_blocks, 2)
    edges, weights = [], []
    for block in range(observed_blocks):
        block_edges = np.quantile(
            pairs[:, block, 0], np.linspace(0.0, 1.0, bins + 1)
        )
        block_edges[0], block_edges[-1] = -np.inf, np.inf
        labels = np.clip(
            np.digitize(pairs[:, block, 0], block_edges[1:-1]),
            0,
            bins - 1,
        )
        edges.append(block_edges)
        weights.append(np.bincount(labels, minlength=bins) / labels.size)
    return edges, weights


def fixed_output_metrics(
    result: RunResult,
    problem: CurvedObservationProblem,
    data_active_reference: np.ndarray,
    bin_edges: list[np.ndarray],
    bin_weights: list[np.ndarray],
    *,
    sliced_directions: int,
    model_name: str,
    sampler_name: str,
    transport_nfe: float,
    parameter_count: int,
    training_time_sec: float,
) -> dict[str, object]:
    """Metrics for a fixed number of retained samples per output chain."""
    if result.data_active_trace is None:
        raise ValueError("fixed-output evaluation requires data_active_trace")
    trace = np.asarray(result.data_active_trace)
    flattened = trace.reshape(-1, problem.active_dim)
    finite_rows = np.all(np.isfinite(flattened), axis=1)
    finite_flattened = flattened[finite_rows]
    finite_fraction = float(np.mean(finite_rows))
    pairs = finite_flattened.reshape(-1, problem.observed_blocks, 2)
    tv = []
    for block in range(problem.observed_blocks):
        if len(pairs):
            labels = np.clip(
                np.digitize(pairs[:, block, 0], bin_edges[block][1:-1]), 0, 7
            )
            observed = np.bincount(labels, minlength=8) / labels.size
            tv.append(0.5 * np.sum(np.abs(observed - bin_weights[block])))
        else:
            tv.append(float("nan"))

    if finite_fraction == 1.0:
        coordinate_values = coordinate_ess(trace)
        rhat = split_rhat(trace)
    else:
        coordinate_values = np.full(problem.active_dim, np.nan)
        rhat = np.full(problem.active_dim, np.nan)
    blocks = trace.reshape(
        trace.shape[0], trace.shape[1], problem.observed_blocks, 2
    )
    observation = problem.observation_values.reshape(1, 1, -1)
    residual = (problem.observe_blocks(blocks) - observation) / problem.sigma_y
    phi_trace = 0.5 * np.sum(residual**2, axis=-1)
    likelihood_ess = (
        chain_ess(phi_trace) if np.all(np.isfinite(phi_trace)) else float("nan")
    )
    runtime = max(float(result.runtime_sec), 1e-12)
    swap = np.asarray(result.swap_acceptance)
    finite_swap = swap[np.isfinite(swap)]
    finite_rhat = rhat[np.isfinite(rhat)]
    finite_tv = np.asarray(tv)[np.isfinite(tv)]
    finite_coordinate_ess = coordinate_values[np.isfinite(coordinate_values)]
    coordinate_mean = (
        float(np.mean(finite_coordinate_ess))
        if len(finite_coordinate_ess) else float("nan")
    )
    coordinate_median = (
        float(np.median(finite_coordinate_ess))
        if len(finite_coordinate_ess) else float("nan")
    )
    coordinate_minimum = (
        float(np.min(finite_coordinate_ess))
        if len(finite_coordinate_ess) else float("nan")
    )
    timing = result.timing_sec or {}
    return {
        "method": result.method,
        "model": model_name,
        "sampler": sampler_name,
        "seed": result.seed,
        "retained_chains": int(result.retained_chains or trace.shape[1]),
        "retained_per_chain": int(result.retained_per_chain or trace.shape[0]),
        "nominal_retained_samples": int(np.prod(trace.shape[:2])),
        "finite_sample_fraction": finite_fraction,
        "data_active_sliced_w2": (
            sliced_w2(
                finite_flattened,
                data_active_reference,
                sliced_directions,
                result.seed + 813,
            )
            if len(finite_flattened) >= 2 else float("nan")
        ),
        "mean_block_bin_tv": (
            float(np.mean(finite_tv)) if len(finite_tv) else float("nan")
        ),
        "data_coordinate_ess_mean": coordinate_mean,
        "data_coordinate_ess_median": coordinate_median,
        "data_coordinate_ess_min": coordinate_minimum,
        "data_coordinate_ess_mean_per_sec": float(coordinate_mean / runtime),
        "log_likelihood_ess": float(likelihood_ess),
        "log_likelihood_ess_per_sec": float(likelihood_ess / runtime),
        "split_rhat_median": (
            float(np.median(finite_rhat)) if len(finite_rhat) else float("nan")
        ),
        "split_rhat_max": (
            float(np.max(finite_rhat)) if len(finite_rhat) else float("nan")
        ),
        "cold_acceptance": float(result.local_acceptance[-1]),
        "cold_hmc_acceptance": float(result.hmc_acceptance[-1]),
        "cold_pcn_acceptance": float(result.pcn_acceptance[-1]),
        "minimum_swap_acceptance": (
            float(np.min(finite_swap)) if len(finite_swap) else float("nan")
        ),
        "runtime_sec": float(result.runtime_sec),
        "adaptation_sec": float(timing.get("adaptation", float("nan"))),
        "burnin_sec": float(timing.get("burnin_after_adaptation", float("nan"))),
        "retained_sampling_sec": float(timing.get("retained_sampling", float("nan"))),
        "final_transport_sec": float(timing.get("final_transport", float("nan"))),
        "phi_evaluations": int(result.phi_evaluations),
        "gradient_evaluations": int(result.gradient_evaluations),
        "transport_nfe_per_call": float(transport_nfe),
        "network_parameters": int(parameter_count),
        "offline_training_sec": float(training_time_sec),
    }


def make_end_to_end_sample_plot(
    path: Path,
    results: list[RunResult],
    labels: list[str],
    data_active_reference: np.ndarray,
    problem: CurvedObservationProblem,
    *,
    max_points: int = 2500,
    block_index: int = 0,
) -> None:
    """Plot one preselected data-coordinate block for all end-to-end methods."""
    if len(results) != len(labels):
        raise ValueError("results and labels must have equal length")
    block = int(block_index)
    if not 0 <= block < problem.observed_blocks:
        raise ValueError("block_index outside the problem")
    reference = data_active_reference.reshape(-1, problem.observed_blocks, 2)[:, block]
    datasets = [("Reference", reference, 1.0)]
    for label, result in zip(labels, results):
        if result.data_active_trace is None:
            continue
        values = result.data_active_trace.reshape(-1, problem.observed_blocks, 2)[:, block]
        finite = np.all(np.isfinite(values), axis=1)
        datasets.append((label, values[finite], float(np.mean(finite))))
    valid = [(label, values, fraction) for label, values, fraction in datasets if len(values)]
    count = min(max_points, *(len(values) for _, values, _ in valid))
    rng = np.random.default_rng(20260901 + block)
    shown = []
    for label, values, fraction in datasets:
        if len(values) > count:
            values = values[rng.choice(len(values), count, replace=False)]
        shown.append((label, values, fraction))
    combined = reference
    lower = np.quantile(combined, 0.0025, axis=0)
    upper = np.quantile(combined, 0.9975, axis=0)
    padding = 0.06 * np.maximum(upper - lower, 1e-6)
    columns = min(5, len(shown))
    rows = int(np.ceil(len(shown) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.1 * columns, 3.8 * rows),
        constrained_layout=True, squeeze=False,
    )
    colors = ["0.2", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for axis, (label, values, fraction), color in zip(axes.flat, shown, colors):
        if len(values):
            axis.scatter(values[:, 0], values[:, 1], s=6, alpha=0.22, color=color, linewidths=0, rasterized=True)
        else:
            axis.text(
                0.5, 0.5, "No finite samples", ha="center", va="center",
                transform=axis.transAxes, color="tab:red",
            )
        axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        axis.set_xlabel(fr"data $x_{{{2 * block + 1}}}$")
        axis.set_ylabel(fr"data $x_{{{2 * block + 2}}}$")
        suffix = "" if fraction == 1.0 else f"; finite={100 * fraction:.1f}%"
        inside = np.all((values >= lower - padding) & (values <= upper + padding), axis=1)
        suffix += f"; in window={100 * np.mean(inside):.1f}%" if len(values) else ""
        axis.set_title(label + suffix)
        axis.grid(alpha=0.18)
    for axis in axes.flat[len(shown):]:
        axis.axis("off")
    fig.suptitle(
        f"{problem.scenario}; m={problem.observed_blocks}; "
        f"sigma_y={problem.sigma_y:g}; c={problem.observation_tilt:g}; "
        f"block {block + 1}; y={problem.observation_values[block]:g}; n={count}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def write_end_to_end_report(
    path: Path,
    *,
    arguments: dict[str, object],
    training_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
) -> None:
    """Write a compact, explicit protocol report for the learned benchmark."""
    lines = [
        "# Fixed-output learned-transport comparison",
        "",
        "## Protocol",
        "",
        "- All d/2 pairs are curved; only the first m blocks enter the likelihood. Prior-fit diagnostics cover all d coordinates; posterior metrics cover the observed data coordinates.",
        "- Every MCMC method returns the same number of retained samples per output chain after a predeclared adaptation and burn-in.",
        "- The number of output chains is matched. SPT hot replicas are auxiliary and their complete cost is included.",
        "- Runtime includes initialization, adaptation, burn-in, rejected proposals, replica swaps, retained sampling, and final source-to-data transport.",
        "- Offline prior-model training is excluded from test-time runtime and reported separately.",
        "- iMF, flow matching, and diffusion use a common feed-forward MLP width/depth, fixed training dataset, batch size, optimizer, and update count; exact parameter counts are reported.",
        "- Flow matching uses its selected coupling and iMF/diffusion use their method-native training objectives.",
        "- All methods use the same explicit tilted observation G_c(x1,x2)=x2+c*x1; c is recorded in the arguments.",
        "- The primary comparison fixes nominal output count; ESS/sec and data-space distributional error expose correlation and approximation error.",
        "",
        "## Arguments",
        "",
        f"```json\n{json.dumps(arguments, indent=2, default=str)}\n```",
        "",
        "## Training summary",
        "",
    ]
    for row in training_rows:
        lines.append(
            f"- {row['kind']}: parameters={int(row['parameter_count'])}, "
            f"training={float(row['training_time_sec']):.3f}s, "
            f"final loss={float(row['final_loss']):.5g}."
        )
    lines.extend(("", "## Fixed-output results", ""))
    for row in metric_rows:
        lines.append(
            f"- {row['method']}: SW2={float(row['data_active_sliced_w2']):.5g}, "
            f"mean ESS={float(row['data_coordinate_ess_mean']):.3f}, "
            f"runtime={float(row['runtime_sec']):.3f}s, "
            f"ESS/s={float(row['data_coordinate_ess_mean_per_sec']):.5g}."
        )
    lines.extend((
        "",
        "Interpret runtime and accuracy jointly. A method that reaches the fixed nominal count quickly can still be inferior if those samples are correlated or target the wrong learned posterior.",
        "",
    ))
    path.write_text("\n".join(lines))



def make_sample_heatmap(
    path: Path,
    pairs: dict[str, np.ndarray],
    *,
    block_index: int,
    title: str,
    bins: int = 64,
    columns: int = 5,
) -> None:
    """Compare 2D histogram densities on common bins and one linear color scale.

    Use every sample, not scatter-plot thinning. Normalize by all finite samples
    and bin area, retaining the probability loss outside the reference window.
    """
    reference = np.asarray(pairs["Reference"]).reshape(-1, 2)
    reference = reference[np.isfinite(reference).all(axis=1)]
    if not len(reference):
        raise ValueError("Heatmaps require at least one finite reference sample")
    lower, upper = np.quantile(reference, [0.0025, 0.9975], axis=0)
    padding = 0.06 * np.maximum(upper - lower, 1e-6)
    lower, upper = lower - padding, upper + padding
    x_edges = np.linspace(lower[0], upper[0], bins + 1)
    y_edges = np.linspace(lower[1], upper[1], bins + 1)
    area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    histograms = []
    for label, raw in pairs.items():
        raw = np.asarray(raw).reshape(-1, 2)
        values = raw[np.isfinite(raw).all(axis=1)]
        counts, _, _ = np.histogram2d(
            values[:, 0], values[:, 1], bins=(x_edges, y_edges)
        )
        density = counts / (len(values) * area) if len(values) else counts
        inside = float(counts.sum() / len(values)) if len(values) else 0.0
        finite = len(values) / len(raw) if len(raw) else 0.0
        histograms.append((label, density, len(values), inside, finite))
    vmax = max(float(item[1].max()) for item in histograms)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    columns = min(columns, len(histograms))
    rows = int(np.ceil(len(histograms) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.3 * columns + 0.7, 3.8 * rows),
        sharex=True, sharey=True, squeeze=False, constrained_layout=True,
    )
    used_axes = []
    for axis, (label, density, count, inside, finite) in zip(axes.flat, histograms):
        mesh = axis.pcolormesh(
            x_edges, y_edges, density.T, cmap="viridis", norm=norm,
            shading="flat", rasterized=True,
        )
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_xlabel(fr"data $x_{{{2 * block_index + 1}}}$")
        axis.set_ylabel(fr"data $x_{{{2 * block_index + 2}}}$")
        #suffix = "" if finite == 1.0 else f"; finite={100 * finite:.1f}%"
        #axis.set_title(f"{label}\nn={count:,}; in window={100 * inside:.1f}%{suffix}", fontsize=10)
        axis.set_title(f"{label}")
        if not count:
            axis.text(0.5, 0.5, "No finite samples", transform=axis.transAxes,
                      ha="center", va="center", color="white")
        used_axes.append(axis)
    for axis in axes.flat[len(histograms):]:
        axis.set_visible(False)
    fig.colorbar(mesh, ax=used_axes, label="Probability density")
    #fig.suptitle(title + f"\n{bins} × {bins} common bins; all samples; no smoothing", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_posterior_heatmap(
    path: Path,
    results: list[RunResult],
    labels: list[str],
    reference: np.ndarray,
    problem: CurvedObservationProblem,
    *,
    block_index: int,
    seed: int,
    bins: int,
) -> None:
    if len(results) != len(labels):
        raise ValueError("results and labels must have equal length")
    pairs = {"Reference": reference.reshape(-1, problem.observed_blocks, 2)[:, block_index]}
    for label, result in zip(labels, results):
        if result.data_active_trace is not None:
            pairs[label] = result.data_active_trace.reshape(
                -1, problem.observed_blocks, 2
            )[:, block_index]
    make_sample_heatmap(
        path, pairs, block_index=block_index, bins=bins,
        title=(f"Posterior: {problem.scenario}; m={problem.observed_blocks}; "
               f"sigma_y={problem.sigma_y:g}; c={problem.observation_tilt:g}; "
               f"block {block_index + 1}; y={problem.observation_values[block_index]:g}; seed={seed}"),
    )


def make_prior_diagnostic_plots(
    outdir: Path,
    reference_active: np.ndarray,
    generated_active: dict[str, np.ndarray],
    *,
    max_points: int,
    heatmap_bins: int = 64,
):
    """Plot first-block prior geometry and marginal empirical CDFs.

    The arrays contain all data coordinates as consecutive pairs.  All panels use the same held-out
    reference sample and deterministic thinning.
    """
    np.savez_compressed(outdir / "prior_samples_raw.npz", reference=reference_active, **generated_active)
    arrays = {"Reference": np.asarray(reference_active)}
    arrays.update({kind: np.asarray(value) for kind, value in generated_active.items()})
    pairs = {
        label: values.reshape(values.shape[0], -1, 2)[:, 0, :]
        for label, values in arrays.items()
    }
    colors = {
        "Reference": "#4c4c4c",
        "imf": "#1f77b4",
        "fm": "#d62728",
        "diffusion": "#9467bd",
    }

    columns = len(pairs)
    figure, axes = plt.subplots(
        1, columns, figsize=(4.1 * columns, 3.8), squeeze=False,
        sharex=True, sharey=True,
    )
    rng = np.random.default_rng(91_337)
    pooled = pairs["Reference"]
    x_limits = np.quantile(pooled[:, 0], (0.002, 0.998))
    y_limits = np.quantile(pooled[:, 1], (0.002, 0.998))
    for axis, (label, values) in zip(axes[0], pairs.items()):
        count = min(int(max_points), len(values))
        indices = rng.choice(len(values), size=count, replace=False)
        axis.scatter(
            values[indices, 0], values[indices, 1], s=7, alpha=0.28,
            color=colors.get(label, None), linewidths=0,
        )
        inside = np.isfinite(values).all(axis=1) & (values[:, 0] >= x_limits[0]) & (values[:, 0] <= x_limits[1]) & (values[:, 1] >= y_limits[0]) & (values[:, 1] <= y_limits[1])
        axis.set_title(f"{label}; in window={100 * np.mean(inside):.1f}%")
        axis.set_xlabel(r"data $x_1$")
        axis.grid(alpha=0.2)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
    axes[0, 0].set_ylabel(r"data $x_2$")
    #figure.suptitle("Held-out prior samples: first active block")
    figure.tight_layout()
    figure.savefig(outdir / "prior_samples.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    make_sample_heatmap(
        outdir / "prior_samples_heatmap.png", pairs, block_index=0,
        title="Held-out prior samples: first active block",
        bins=heatmap_bins, columns=len(pairs),
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for coordinate, axis in enumerate(axes):
        for label, values in pairs.items():
            ordered = np.sort(values[:, coordinate])
            probability = (np.arange(len(ordered)) + 0.5) / len(ordered)
            axis.plot(
                ordered, probability, label=label,
                color=colors.get(label, None), linewidth=1.7,
                linestyle="--" if label == "Reference" else "-",
            )
        axis.set_xlabel(rf"data $x_{coordinate + 1}$")
        axis.set_ylabel("empirical CDF")
        axis.set_xlim(*(x_limits if coordinate == 0 else y_limits))
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.suptitle("Held-out prior marginals: first active block")
    figure.tight_layout()
    figure.savefig(outdir / "prior_marginals.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
