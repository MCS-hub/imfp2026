# Branches

This repo consists of two braches:

- Branch main: synthetic posterior-sampling experiments
- Branch image: CLIP-guided Imagenet experiments


# Synthetic experiments (this brach)

## 1. Environment

Use Python 3.10 or newer and PyTorch 2.0 or newer. Install a PyTorch build
appropriate for your hardware (CPU, NVIDIA CUDA, or AMD ROCm), then install the
remaining dependencies with:

```bash
python -m pip install -r requirements.txt
```

Use `--device cpu` to force CPU or `--device cuda` to require a GPU. PyTorch uses
the device name `cuda` for both CUDA and ROCm builds. A GPU is recommended for the
full posterior-sampling comparison.

## 2. Prior pretrained checkpoints

Those checkpoints are given in folder 'checkpoints'

## 3. Main posterior-sampling experiments

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

## 4. Two-dimensional loss-to-map-error experiment

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

## 5. Quick execution checks

These reduced runs check execution only. Use separate checkpoint and output directories for them.

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

