# Synthetic experiments: posterior sampling with few-step transports

This repository contains the synthetic posterior-sampling experiments and the
2D banana loss-to-map-error experiment accompanying the manuscript. All datasets
and numerical posterior references are generated locally; no external datasets
or pretrained models are required. Run the commands below from the repository
root in an activated Python environment. The cluster-specific shell scripts are
not needed.

## 1. Environment

Use Python 3.10 or newer and PyTorch 2.0 or newer. Install a PyTorch build
appropriate for your hardware (CPU, NVIDIA CUDA, or AMD ROCm), then install the
remaining dependencies with:

```bash
python -m pip install -r requirements.txt
```

The dependencies are PyTorch, NumPy, SciPy, and Matplotlib. `torchdiffeq` is
optional and is not needed for the reported experiments, which use fixed-step
RK4 for flow matching. The recorded 2D run used PyTorch 2.9.1 with ROCm 6.4 and
NumPy 2.2.6 on an AMD Instinct MI250X.

Both experiment drivers select an available GPU automatically, otherwise CPU.
Use `--device cpu` to force CPU or `--device cuda` to require a GPU. PyTorch uses
the device name `cuda` for both CUDA and ROCm builds. A GPU is recommended for the
full posterior-sampling comparison. Figures are generated without a display.

## 2. Main posterior-sampling experiments

Run the banana experiment:

```bash
python -u end_to_end_comparison.py \
  --scenario banana \
  --dim 32 \
  --observed-blocks 16 \
  --checkpoint-dir checkpoints \
  --outdir outs/banana
```

Run the sine experiment:

```bash
python -u end_to_end_comparison.py \
  --scenario sine \
  --dim 32 \
  --observed-blocks 16 \
  --checkpoint-dir checkpoints \
  --outdir outs/sine
```

The explicit `--dim 32` is important: the driver's general-purpose default is 64.
Each command trains the prior models if compatible checkpoints are unavailable,
evaluates prior quality, constructs a numerical posterior reference, and runs:

| Code identifier | Method |
| --- | --- |
| `imf_spt_pcn` | iMF transport with SPT and pCN updates |
| `imf_spt_hybrid` | iMF transport with SPT, hot-level pCN, and cold-level split HMC |
| `fm_spt_pcn` | Flow-matching transport with SPT and pCN updates |
| `diffusion_dps` | Diffusion posterior sampling (DPS) |

The commands use the manuscript settings through the following defaults:

| Setting | Value |
| --- | --- |
| Prior training dataset | 20,000 samples per scenario |
| Training | 16,000 updates; batch size up to 512; learning rate 0.001 |
| Network | Four hidden SiLU layers of width 512 |
| Training-data seed | 1701 |
| Observation noise standard deviation | 0.2 |
| Observation operator | `G(x1, x2) = x2 + 0.35*x1` in each block |
| Fixed observations | First two blocks: 2; remaining blocks: -1 |
| Banana / sine transverse standard deviation | 0.30 / 0.15 |
| iMF generation | Six steps |
| FM generation | 100 RK4 steps, or 400 network evaluations per map |
| SPT | Ten ladders, each with 24 inverse temperatures |
| Adaptation / additional burn-in | 500 / 500 sweeps |
| Retained samples | 600 per ladder, without thinning: 6,000 per method and seed |
| Cold-level split HMC | Six integration steps; initial step size 0.05 |
| DPS | 1,000 diffusion steps; residual-norm guidance scale 0.3 |
| Sampler seeds | 0, 1, 2, 3 |

Prior models are fixed across sampler seeds. The reported standard deviations
therefore describe sampler-seed variability, conditional on the fitted models.
The 6,000 retained samples exclude adaptation and burn-in; reported sampling
runtime includes these phases, initialization, swaps, and final transport.

### Expected outputs

Each scenario's output directory contains:

| File | Contents |
| --- | --- |
| `summary.csv` | Mean and sample standard deviation across sampler seeds; source of the synthetic results table |
| `by_seed.csv` | Individual-seed errors, ESS, runtime, acceptance rates, and split-Rhat diagnostics |
| `training.csv` | Model sizes, training costs, losses, and training metadata |
| `prior_diagnostics.csv` | Learned-prior distributional diagnostics |
| `protocol.json` | Resolved experiment arguments and protocol metadata |
| `REPORT.md` | Automatically generated run summary |
| `prior_samples_heatmap.png` | Reference and learned-prior histograms |
| `posterior_leading_observation_heatmap.png` | Posterior histograms for a block observed at 2 |
| `posterior_remaining_observation_heatmap.png` | Posterior histograms for a block observed at -1 |

Additional scatter plots, marginal plots, and prior sample arrays are saved.
Posterior figures use sampler seed 0. Model checkpoints are written to
`--checkpoint-dir`.

The principal CSV metrics are `data_active_sliced_w2` (sliced Wasserstein),
`mean_block_bin_tv` (mean first-coordinate binned TV across blocks),
`data_coordinate_ess_mean` (coordinate ESS),
`data_coordinate_ess_mean_per_sec` (coordinate ESS/s), and `runtime_sec`.
Aggregate fields in `summary.csv` append `_mean` and `_std` to these names.
DPS ESS and MCMC acceptance/convergence diagnostics are marked `NaN`, since DPS
outputs come from independently generated trajectories rather than an MCMC chain.

### Runtime and checkpoint reuse

The saved main-benchmark sampling times total approximately **8–9 hours per
scenario** across all four methods and four seeds, dominated by the FM baseline.
This excludes prior training, reference construction, and evaluation. Runtime
depends on hardware and software; CPU execution can be substantially slower.

Compatible prior checkpoints are reused by default. To retrain, append
`--no-reuse-checkpoints`, or select a fresh checkpoint directory. Changing a
training configuration while reusing incompatible checkpoints raises an error.
For an independent reproduction, use fresh checkpoint and output directories.
The main driver can overwrite existing output files.

## 3. Two-dimensional loss-to-map-error experiment

Run the experiment supporting the loss-to-map-error result:

```bash
python -u banana_map_error_2d.py \
  --seeds 0 \
  --outdir outs/banana_map_error_2d
```

The default is one training seed, 20,000 prior samples, 16,000 optimizer updates,
and three hidden SiLU layers of width 128. Evaluation occurs at initialization
and every 500 updates using 8,192 held-out risk samples and 2,048 Gaussian source
samples. Both one- and six-step maps are evaluated. The GPU training run takes
approximately a few minutes; this is an estimate rather than a saved timing
benchmark.

The reference map is obtained by integrating the **marginal flow for independent
interpolation**, not by using the triangular banana sampling map. Its velocity
is evaluated by 128-point Gaussian quadrature and its flow by a double-precision
DOP853 solver. The script checks sensitivity to doubled quadrature order and
tighter ODE tolerances before training, and stops if its accuracy check fails.

The output directory contains:

- `metrics.csv`: checkpoint-wise marginal risk, paired map error, Monte Carlo
  standard errors, and sampled spatial-Jacobian diagnostics.
- `reference_checks.json` and `reference.npz`: reference-accuracy checks and
  fixed evaluation samples/reference values.
- `config.json`: resolved configuration and completion status.
- `imf_seed0_update16000.pt`: the final model. Add `--save-checkpoints` to save
  intermediate evaluated checkpoints as well.
- `risk_vs_map_error.png` and `.pdf`: the manuscript's risk-versus-map-error plot.
- `risk_vs_training.png` and `.pdf`: marginal risk and raw loss versus updates.
- `spatial_jacobian_vs_training.png` and `.pdf`: the sampled regularity diagnostic.
- `plot_config.json`: the selected plotting run and figure settings.

`delta` is the directly estimated marginal joint risk; `map_mse` is the mean
squared paired transport-map error and `map_l2` its square root. These metrics
sum squared errors over coordinates. Error bars are conditional Monte Carlo
standard errors, not variation across training runs. This experiment illustrates
the loss–error relationship; it does not certify the theorem's constants or rate.

This driver requires an empty output directory for a new training run. If saved
results are already present, choose a new directory, for example
`--outdir outs/banana_map_error_2d_reproduction`.

Regenerate the separate figures from existing results without retraining:

```bash
python banana_map_error_2d.py \
  --plot-only \
  --plot-seed 0 \
  --font-size 18 \
  --outdir outs/banana_map_error_2d
```

Plot regeneration replaces the figures but preserves metrics and model files.

## 4. Quick execution checks

These reduced runs check execution only; they do not reproduce the manuscript's
numerical results. Use separate checkpoint and output directories for them.

```bash
python -u end_to_end_comparison.py \
  --quick --device cpu \
  --scenario banana --dim 32 --observed-blocks 16 \
  --checkpoint-dir checkpoints_smoke \
  --outdir outs/banana_smoke

python -u banana_map_error_2d.py \
  --quick --device cpu \
  --outdir outs/map_error_smoke
```

The 2D reference and differentiation checks can also be run independently:

```bash
python banana_map_error_2d.py --self-test
```

For all configurable options:

```bash
python end_to_end_comparison.py --help
python banana_map_error_2d.py --help
```

Random seeds and resolved settings are recorded with the results. Numerical
results need not match bit-for-bit across different hardware or library versions.

## 5. Source files

Keep these files together for the main benchmark:

- `end_to_end_comparison.py`: training, sampling, and evaluation driver.
- `curved_problem.py`: priors, observation models, and posterior references.
- `learned_models.py`: iMF, FM, and diffusion models and DPS sampling.
- `spt_score.py`: pCN, split-HMC, and parallel-tempering implementations.
- `evaluation.py`: metrics, plots, and reports.

`banana_map_error_2d.py` is standalone and needs only the dependencies above.
`test_benchmark.py` contains additional benchmark tests. This repository's
instructions cover the synthetic experiments; they do not run the ImageNet
experiments.
