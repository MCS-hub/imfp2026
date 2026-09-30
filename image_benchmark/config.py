import copy
import csv
import json
import math
from pathlib import Path

DEFAULT = {
    "finite_difference_check": False,
    "device": "cuda", "image_size": 256, "methods": ["imf_spt_pcn", "imf_spt_hybrid", "diffusion_dps"],
    "tasks": ["super_resolution", "phase_retrieval"], "sampler_seeds": [0,1,2,3],
    "observation_seed": 29, "images_csv": "images.csv", "outdir": "outs/image_final",
    "likelihood_batch_size": 1, "save_source_trace": True, "examples": 8,
    "imf": {"checkpoint": "checkpoints/iMF-B-2.pth", "architecture": "imfDiT_B_2",
            "steps": 1, "omega": 8.0, "guidance_interval": [0.4,0.65],
            "vae": "stabilityai/sd-vae-ft-mse", "local_files_only": False,
            "activation_checkpointing": False},
    "spt": {"replicas": 24, "chains": 4, "adapt_sweeps": 500, "burnin_sweeps": 500,
            "retained_per_chain": 512, "thin": 1, "temperature_power": 2.0, "betas": None,
            "hmc_steps": 6, "initial_hmc_epsilon": 0.001, "min_hmc_epsilon": 1e-7,
            "max_hmc_epsilon": 0.5, "initial_pcn_scale": 0.01,
            "target_hmc_acceptance": 0.75, "target_pcn_acceptance": 0.30, "log_every": 10},
    "dps": {"checkpoint": "checkpoints/256x256_diffusion.pt", "class_cond": True,
            "steps": 1000, "batch_size": 1,
            "guidance_scale": {"super_resolution": 1.0, "phase_retrieval": 1.0}},
    "observations": {"super_resolution": {"factor": 4, "sigma": 0.05},
                     "phase_retrieval": {"oversample": 2.0, "epsilon": 1e-6, "sigma": 0.05}},
}


def merge(base, update):
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown config key: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            merge(base[key], value)
        else:
            base[key] = value
    return base


def read_config(path):
    with Path(path).open() as f:
        return merge(copy.deepcopy(DEFAULT), json.load(f))


def validate(config):
    c, s = config, config["spt"]
    if type(c["finite_difference_check"]) is not bool:
        raise ValueError("finite_difference_check must be true or false")
    if c["image_size"] != 256:
        raise ValueError("The supplied checkpoints require image_size=256")
    for key, allowed in [("methods", DEFAULT["methods"]), ("tasks", DEFAULT["tasks"])]:
        if not c[key] or any(x not in allowed for x in c[key]) or len(set(c[key])) != len(c[key]):
            raise ValueError(f"Invalid {key}")
    if not c["sampler_seeds"] or any(type(x) is not int or x<0 for x in c["sampler_seeds"]) or len(set(c["sampler_seeds"])) != len(c["sampler_seeds"]):
        raise ValueError("sampler_seeds must be distinct nonnegative integers")
    for key in ["replicas","chains","retained_per_chain","thin","hmc_steps","log_every"]:
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f"spt.{key} must be a positive integer")
    if s["replicas"] < 2 or s["chains"] < 2:
        raise ValueError("At least two replicas and two independent ladders/chains are required")
    for key in ["adapt_sweeps", "burnin_sweeps"]:
        if type(s[key]) is not int or s[key] < 0:
            raise ValueError(f"spt.{key} must be a nonnegative integer")
    for value in [s["initial_pcn_scale"], s["target_hmc_acceptance"], s["target_pcn_acceptance"]]:
        if not 0 < value < 1:
            raise ValueError("pCN scale and target acceptances must lie in (0,1)")
    if not 0 < s["min_hmc_epsilon"] <= s["initial_hmc_epsilon"] <= s["max_hmc_epsilon"] < math.pi:
        raise ValueError("Invalid HMC step-size bounds")
    if not math.isfinite(s["temperature_power"]) or s["temperature_power"] <= 0:
        raise ValueError("temperature_power must be positive")
    if s["betas"] is not None:
        betas = s["betas"]
        if len(betas) != s["replicas"] or betas[0] != 0 or betas[-1] != 1 or not all(a<b for a,b in zip(betas,betas[1:])):
            raise ValueError("betas must be strictly increasing from 0 to 1 with length replicas")
    if c["likelihood_batch_size"] < 1 or c["dps"]["batch_size"] < 1 or c["examples"] < 1:
        raise ValueError("Batch sizes and examples must be positive")
    if not 2 <= c["dps"]["steps"] <= 1000:
        raise ValueError("DPS steps must be between 2 and 1000; the trained schedule stays at 1000")
    if type(c["dps"]["class_cond"]) is not bool:
        raise ValueError("dps.class_cond must be true or false and match the checkpoint")
    imf = c["imf"]
    if imf["steps"] < 1 or imf["omega"] <= 0 or not 0 <= imf["guidance_interval"][0] <= imf["guidance_interval"][1] <= 1:
        raise ValueError("Invalid iMF steps or guidance settings")
    for task in c["tasks"]:
        if not math.isfinite(c["observations"][task]["sigma"]) or c["observations"][task]["sigma"] <= 0:
            raise ValueError("Observation sigma must be positive and finite")
        if c["dps"]["guidance_scale"][task] <= 0:
            raise ValueError("DPS guidance must be positive")


def load_manifest(path):
    path = Path(path).resolve()
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError("The image manifest is empty")
    seen = set()
    for row in rows:
        if not {"image_id", "image_path", "class_id"} <= row.keys():
            raise ValueError("Manifest columns: image_id,image_path,class_id")
        key = row["image_id"]
        if not key or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in key) or key in seen:
            raise ValueError("image_id must be unique and contain only letters, digits, '-' or '_'")
        seen.add(key)
        row["class_id"] = int(row["class_id"])
        if not 0 <= row["class_id"] < 1000:
            raise ValueError("class_id must use the checkpoint's zero-based ImageNet class order")
        p = Path(row["image_path"]).expanduser()
        row["image_path"] = str((path.parent/p).resolve() if not p.is_absolute() else p)
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
    return rows
