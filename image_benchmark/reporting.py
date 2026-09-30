import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float,np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(k for row in rows for k in row))
    temporary = Path(str(path)+".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    result = []
    excluded = set(group_keys) | {"seed", "class_id"}
    for key, items in sorted(groups.items()):
        row = dict(zip(group_keys, key))
        row["runs"] = len(items)
        for name in dict.fromkeys(k for item in items for k in item):
            if name in excluded:
                continue
            values = [item.get(name) for item in items]
            numeric = [v for v in values if isinstance(v,(int,float)) and not isinstance(v,bool)]
            if not numeric:
                continue
            finite = np.asarray([v for v in numeric if math.isfinite(v)], dtype=float)
            row[name+"_valid_runs"] = len(finite)
            if len(finite) != len(items):
                # Do not silently average away failed/undefined chain diagnostics.
                row[name+"_mean"], row[name+"_std"] = None, None
            else:
                row[name+"_mean"] = float(finite.mean())
                row[name+"_std"] = float(finite.std(ddof=1)) if len(finite)>1 else None
        result.append(row)
    return result


def refresh_report(outdir):
    outdir = Path(outdir)
    rows = [json.loads(p.read_text())["metrics"] for p in sorted(outdir.glob("runs/*/*/*/seed_*/result.json"))]
    write_csv(outdir/"by_seed.csv", rows)
    summaries = aggregate(rows, ["task", "method"])
    write_csv(outdir/"summary.csv", summaries)
    write_csv(outdir/"by_image.csv", aggregate(rows, ["task", "image_id", "method"]))
    lines = ["# Image posterior comparison", "", "Means ± sample standard deviations over completed image–sampler-seed runs. "
             "Use by_image.csv for variation over seeds within each image. These runs are conditional on fixed checkpoints; "
             "pooled run standard deviations are not confidence intervals over independently trained models.", "",
             "| Task | Method | Runs | Latent ESS/s | Log-likelihood ESS/s | Time (s) | Mean-image PSNR |",
             "|---|---|---:|---:|---:|---:|---:|"]
    def cell(row, metric):
        mean, std = row.get(metric+"_mean"), row.get(metric+"_std")
        if mean is None:
            return "N/A"
        return f"{mean:.3g}" + (f" ± {std:.2g}" if std is not None else "")
    for row in summaries:
        values = [cell(row,k) for k in ["latent_coordinate_ess_mean_per_sec", "log_likelihood_ess_per_sec", "runtime_sec", "posterior_mean_psnr"]]
        lines.append(f"| {row['task']} | {row['method']} | {row['runs']} | " + " | ".join(values) + " |")
    lines += ["", "ESS: the synthetic code's initial-positive-sequence estimator summed over separate cold ladders, "
              "with no flattening across chain boundaries. ESS and split R-hat are not computed for DPS reverse-time paths. "
              "DPS output rate is nominal independent trajectories per second and does not certify posterior accuracy.", "",
              "Runtime includes initialization, adaptation, burn-in, all replica proposals and swaps, retained sampling, "
              "and rendering/summary accumulation of every retained output. Checkpoint loading, observation preparation, "
              "preflight checks, file writing and final ESS computation are reported separately or excluded.", "",
              "All methods receive the same known ImageNet class ID. The two iMF arms share an identical fixed "
              "transport and pixel likelihood; cross-family results still combine different learned priors and inference procedures. "
              "Phase-retrieval PSNR is unaligned and is sensitive to measurement ambiguities; inspect samples and residuals as well."]
    (outdir/"REPORT.md").write_text("\n".join(lines)+"\n")


def rgb_image(values):
    values = np.clip((np.asarray(values).transpose(1,2,0)+1)/2, 0, 1)
    return Image.fromarray(np.round(values*255).astype(np.uint8))


def save_images(path, truth, measurement, task, arrays):
    path = Path(path)
    rgb_image(arrays["mean"]).save(path/"posterior_mean.png")
    np.save(path/"pixel_std.npy", arrays["std"])
    std = arrays["std"].mean(0)
    maximum = max(float(std.max()), 1e-12)
    Image.fromarray(np.round(255*std/maximum).astype(np.uint8)).save(path/"pixel_std_preview.png")
    tiles = [("Original", rgb_image(truth))]
    if task == "super_resolution":
        observation = rgb_image(measurement).resize((256,256), Image.Resampling.NEAREST)
        title = "Noisy block averages"
    else:
        value = np.log1p(np.abs(measurement).mean(0))
        value = value / max(float(value.max()), 1e-12)
        observation = Image.fromarray(np.round(255*value).astype(np.uint8)).convert("RGB").resize((256,256))
        title = "Noisy log-magnitude preview"
    tiles += [(title, observation), ("Posterior mean", rgb_image(arrays["mean"]))]
    for i, example in enumerate(arrays["examples"]):
        rgb_image(example).save(path/f"sample_{i:02d}.png")
        tiles.append((f"Posterior draw {i+1}", rgb_image(example)))
    columns = min(4,len(tiles))
    grid = Image.new("RGB", (columns*256, math.ceil(len(tiles)/columns)*278), "white")
    draw = ImageDraw.Draw(grid)
    for k,(title,tile) in enumerate(tiles):
        x,y = (k%columns)*256, (k//columns)*278
        draw.text((x+5,y+4), title, fill="black")
        grid.paste(tile.resize((256,256)), (x,y+22))
    grid.save(path/"posterior_panel.png")
