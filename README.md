# Branches

This repo consists of two braches:

- Branch main: synthetic posterior-sampling experiments
- Branch image: CLIP-guided Imagenet experiments

# CLIP-Guided ImageNet Sampling (this branch)

This repository reproduces class-conditional $256x256$ ImageNet experiments
with frozen iMF and SiT transports, a frozen CLIP reward, and sequential
parallel tempering (SPT). The final configurations compare iMF-SPT-pCN,
iMF-SPT-hybrid, an iMF Best-of-12 prior baseline, and a time-budgeted
SiT-SPT-pCN baseline.

## Requirements

Use Python 3.10 or newer with a CUDA-capable PyTorch installation appropriate
for the target GPU. Install the remaining runtime packages:

```bash
python3 -m pip install diffusers transformers safetensors numpy pillow
```

The repository includes the required vendor source trees.

## Downloading checkpoints

Model weights are not included in the submission. From the repository root,
download the two checkpoints below (approximately 5.2 GB combined). These
commands pin the source revisions and save the filenames used by the supplied
configurations:

```bash
mkdir -p checkpoints

curl --fail --location --retry 3 \
  'https://huggingface.co/Lyy0725/iMF/resolve/fbc11eb6687c5b6cff9beec1e6bb28fd2be46adb/iMF-XL-2.pth' \
  --output checkpoints/iMF-XL-2.pth

curl --fail --location --retry 3 \
  'https://huggingface.co/sairights/sit-xl-2-256x256-sde-cfg/resolve/5a15dccbe5710eb5bc9a0b0f71e6fb3deb31e24e/pretrained_models/SiT-XL-2-256x256.safetensors' \
  --output checkpoints/SiT-XL-2-256x256.safetensors
```

Sources:

- **iMF-XL/2:** the [upstream iMF checkpoint repository](https://huggingface.co/Lyy0725/iMF).
- **SiT-XL/2, 256×256:** the [third-party safetensors mirror](https://huggingface.co/sairights/sit-xl-2-256x256-sde-cfg/tree/5a15dccbe5710eb5bc9a0b0f71e6fb3deb31e24e/pretrained_models)
  used for these experiments. The official download was unavailable when the
  checkpoint was acquired. Equivalence to the original official weights has
  not been independently verified; use this exact mirror file to reproduce
  the experiment. Provenance and validation details are recorded in
  [the download metadata](checkpoints/SiT-XL-2-256x256.download.json).

Verify the downloads against the SHA-256 hashes of the experiment checkpoints:

```bash
sha256sum --check <<'EOF'
8a0b02d2fbd640e3a6d045d088946ef300314460a7e5d43bcad15d086095da67  checkpoints/iMF-XL-2.pth
b48cca85a5bf402c0c5039984959dc99ab4d27d59145a4c4cd3b49e1cc0ab660  checkpoints/SiT-XL-2-256x256.safetensors
EOF
```

Both files should report `OK`. The supplied configurations expect these exact,
case-sensitive relative paths. To store weights elsewhere or use different
filenames, edit `imf.checkpoint` and `sit.checkpoint` in the chosen JSON
configuration before starting a new run. Paths are resolved from the working
directory, so run the commands from the repository root. Renaming a checkpoint
does not convert its format: iMF uses PyTorch `.pth` weights and SiT uses
`.safetensors`. The final comparison requires both checkpoints, even though
SiT runs last.

The Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`) and CLIP
(`openai/clip-vit-base-patch32`) weights are downloaded automatically from
Hugging Face on first use unless already cached. Internet access is required
for those initial downloads; offline runs require a populated Hugging Face
cache.

## Running a final configuration

From the repository root, first validate paths and settings without loading
models:

```bash
python3 run_final_guidance.py \
  --config configs/reward_final_golden_retriever_autumn_running_tau200_r8_l8_v1.json \
  --dry-run
```

Run the same configuration with the portable launcher:

```bash
bash scripts/run_reward_final.sh \
  configs/reward_final_golden_retriever_autumn_running_tau200_r8_l8_v1.json
```

For a Slurm system, submit the launcher while supplying the configuration as
its argument, for example:

```bash
sbatch --gpus=1 --cpus-per-task=8 --mem=64G --time=3-00:00:00 \
  --wrap='bash scripts/run_reward_final.sh configs/reward_final_golden_retriever_autumn_running_tau200_r8_l8_v1.json'
```

The launcher uses the active Python interpreter. Set `PYTHON=/path/to/python`
to select another environment and `OMP_NUM_THREADS` to choose the CPU-thread
limit. It writes results beneath the configuration's `outs/` directory.

## Final protocol

Each final configuration uses one prompt and seed, eight temperature replicas,
and eight independent ladders. The iMF methods use 250 adaptation sweeps, 250
burn-in sweeps, and 1,000 retained sweeps per ladder. The Best-of-12 baseline
draws 12 independent iMF prior samples for each of 8,000 output slots and
retains the highest-CLIP-reward candidate in each group, for 96,000 prior
candidates total. iMF-SPT-hybrid uses six-step split HMC at the cold replica
and pCN at the remaining replicas. SiT is run after the iMF methods with a
per-prompt sampling budget equal to the measured iMF-SPT-hybrid sampling time,
plus one hour for loading, preflight, scoring, and export.

Experiment outputs record the resolved protocol, checkpoint hashes, timing,
diagnostics, traces, and rendered samples. Use `--resume` only with an
unchanged configuration and output directory.
