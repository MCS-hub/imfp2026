# Branches

This repo consists of two braches:

- Branch main: synthetic posterior-sampling experiments
- Branch image: CLIP-guided Imagenet experiments (https://github.com/MCS-hub/imfp2026/tree/image)


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

Those checkpoints (for dimension 32) are given in folder `checkpoints`.

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

## 4. Two-dimensional loss-to-map-error experiment

Run the experiment supporting the loss-to-map-error result:

```bash
python -u banana_map_error_2d.py \
  --seeds 0 \
  --outdir outs/banana_map_error_2d
```


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


