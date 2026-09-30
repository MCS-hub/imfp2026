import gc
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from .models import ImageProblem, load_imf
from .observations import make_operator, PixelLikelihood
from .samplers import ImageSPT, synchronize
from .diagnostics import ImageAccumulator, MCMC_FIELDS, mcmc_diagnostics
from .dps import load_dps, sample_dps
from .reporting import write_json, refresh_report, save_images
from .vendor import ROOT


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ")+message, flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_image(path, size=256):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        scale = size/min(img.size)
        img = img.resize((round(img.width*scale),round(img.height*scale)), Image.Resampling.BICUBIC)
        left, top = (img.width-size)//2, (img.height-size)//2
        img = img.crop((left,top,left+size,top+size))
        return torch.from_numpy(np.array(img, dtype=np.float32).transpose(2,0,1)/127.5-1).unsqueeze(0)


def make_likelihood(job, config, device):
    operator = make_operator(job["task"], config["observations"][job["task"]]).to(device)
    return PixelLikelihood(operator, job["measurement"].to(device), config["observations"][job["task"]]["sigma"])


def prepare_jobs(config, manifest, outdir, resume):
    """Observations are drawn once on CPU, independent of methods and sampler seeds."""
    jobs = []
    for image in manifest:
        truth = read_image(image["image_path"])
        for task in config["tasks"]:
            directory = outdir/"observations"/task/image["image_id"]
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory/"observation.npz"
            digest = hashlib.sha256(f"{config['observation_seed']}:{image['image_id']}:{task}".encode()).digest()
            observation_seed = int.from_bytes(digest[:8], "little") % (2**63-1)
            if resume and destination.is_file():
                with np.load(destination) as saved:
                    if not np.array_equal(saved["truth"], truth.numpy()):
                        raise ValueError("Preprocessed image changed; use a new output directory")
                    measurement = torch.from_numpy(saved["measurement"].copy())
            else:
                with torch.no_grad():
                    noiseless = make_operator(task,config["observations"][task])(truth)
                    rng = torch.Generator().manual_seed(observation_seed)
                    noise = torch.randn(noiseless.shape, generator=rng)
                    measurement = noiseless + config["observations"][task]["sigma"]*noise
                np.savez_compressed(destination, truth=truth.numpy(), measurement=measurement.numpy(),
                                    noiseless=noiseless.numpy(), observation_seed=np.array(observation_seed))
            jobs.append({**image, "task":task, "truth":truth, "measurement":measurement})
    return jobs


def directional_gradient_check(fn, position, gradient, tolerance=0.01):
    """Find two neighboring finite-difference scales agreeing with autograd.

    Few-step composition can require much smaller perturbations than one-step
    generation. Conversely, float32 cancellation can spoil the smallest scales.
    Keep the tolerance fixed and require two consecutive successful checks.
    """
    direction = gradient / gradient.norm()
    analytic = (gradient.double() * direction.double()).sum()
    checks = []
    steps = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4,
             1e-4, 5e-5, 3e-5, 2e-5, 1e-5, 3e-6)
    for step in steps:
        with torch.no_grad():
            plus = fn(position + step * direction).double()
            minus = fn(position - step * direction).double()
            finite_difference = (plus - minus) / (2 * step)
        relative = float((finite_difference-analytic).abs() / analytic.abs().clamp_min(1e-12))
        checks.append({"step": step, "finite_difference": float(finite_difference),
                       "relative_error": relative})
        print(f"step={step:.1e}, finite_difference={float(finite_difference):.9e}, "
              f"relative_error={relative:.6g}", flush=True)
        if len(checks) >= 2 and all(
                np.isfinite(c["relative_error"]) and c["relative_error"] < tolerance
                for c in checks[-2:]):
            return {"directional_derivative_relative_error": relative,
                    "directional_derivative_analytic": float(analytic),
                    "directional_derivative_tolerance": tolerance,
                    "directional_derivative_accepted_steps": [c["step"] for c in checks[-2:]],
                    "directional_derivative_checks": checks}
    raise RuntimeError(
        "Preflight gradient check failed: no two neighboring finite-difference "
        f"step sizes agreed with autograd within {tolerance:.1%}. "
        "Inspect the step-size sweep or run scripts/diagnose_imf_gradient.py."
    )


def preflight(problem, finite_difference_check=False):
    """Check finite gradients and deterministic values; finite differences are opt-in."""
    rng = torch.Generator(device=problem.device).manual_seed(732)
    z = torch.randn(1,problem.dim, generator=rng, device=problem.device, dtype=problem.dtype)
    value, gradient = problem.phi_and_grad(z)
    with torch.no_grad():
        again = problem.phi_torch(z)
        latent, image = problem.render(z)
    if not torch.isfinite(value).all() or not torch.isfinite(gradient).all() or float(gradient.norm()) == 0:
        raise RuntimeError("Preflight failed: nonfinite or identically zero likelihood gradient")
    torch.testing.assert_close(value, again, rtol=1e-6, atol=1e-4)
    print(f"phi={float(value):.9e}, gradient_norm={float(gradient.norm()):.9e}", flush=True)
    check = {"finite_difference_check": bool(finite_difference_check)}
    if finite_difference_check:
        check.update(directional_gradient_check(problem.phi_torch, z, gradient))
    else:
        log("Finite-difference gradient check disabled")
    return {"phi":float(value), "gradient_norm":float(gradient.norm()), **check,
            "latent_shape":list(latent.shape), "pixel_shape":list(image.shape)}


def run_imf(problem, config, method, seed):
    result = ImageSPT(problem,config["spt"],method,seed).run(progress=log)
    source = result.pop("source")
    draws, chains, dim = source.shape
    total = draws*chains
    indices = np.linspace(0,total-1,min(total,config["examples"]),dtype=int)
    accumulator = ImageAccumulator(problem.truth,problem.likelihood,indices)
    latents = np.empty((total,dim),dtype=np.float32)
    synchronize(problem.device)
    begin = time.perf_counter()
    for first in range(0,total,problem.batch_size):
        batch = source.reshape(total,dim)[first:first+problem.batch_size].to(problem.device)
        with torch.no_grad():
            latent, image = problem.render(batch)
            latents[first:first+len(batch)] = latent.flatten(1).cpu().numpy()
            accumulator.consume(image)
    image_metrics, arrays = accumulator.finish()
    synchronize(problem.device)
    render_sec = time.perf_counter()-begin
    runtime = result["sampling_sec"]+render_sec
    # The cached likelihood must follow the state through every replica swap.
    np.testing.assert_allclose(arrays["phi"],result["phi"].reshape(-1),rtol=1e-5,atol=1e-4)
    diagnostics_start = time.perf_counter()
    latent_trace = latents.reshape(draws,chains,dim)
    log_likelihood = -result["phi"]
    diagnostics, coordinate_ess, coordinate_rhat = mcmc_diagnostics(
        latent_trace,log_likelihood,runtime,arrays["image_summaries"].reshape(draws,chains,-1))
    metrics = {**image_metrics, **diagnostics, "runtime_sec":runtime,
               "sampling_sec":result["sampling_sec"], "final_render_sec":render_sec,
               "diagnostics_sec":time.perf_counter()-diagnostics_start,
               "nominal_samples":total, "samples_per_sec":total/runtime,
               "cold_acceptance":result["kernel"]["local_acceptance"][-1],
               "minimum_swap_acceptance":min(result["kernel"]["swap_acceptance"]),
               "phi_evaluations":result["kernel"]["phi_evaluations"],
               "gradient_evaluations":result["kernel"]["gradient_evaluations"]}
    traces = {"latent":latent_trace, "log_likelihood":log_likelihood,
              "image_summaries":arrays["image_summaries"].reshape(draws,chains,-1),
              "coordinate_ess":coordinate_ess, "coordinate_split_rhat":coordinate_rhat}
    if config["save_source_trace"]:
        traces["source"] = source.numpy()
    result.pop("phi")
    result["timing"]["final_render"] = render_sec
    return metrics, arrays, traces, result


def run_dps(model, sampler, job, likelihood, config, seed, device):
    total = config["spt"]["chains"]*config["spt"]["retained_per_chain"]
    indices = np.linspace(0,total-1,min(total,config["examples"]),dtype=int)
    accumulator = ImageAccumulator(job["truth"].to(device),likelihood,indices)
    synchronize(device)
    start = time.perf_counter()
    sample_dps(model,sampler,likelihood,total,config["dps"]["batch_size"],seed,
               (3,256,256),device,config["dps"]["guidance_scale"][job["task"]],
               job["class_id"],config["dps"]["class_cond"],
               accumulator.consume,progress=log)
    metrics, arrays = accumulator.finish()
    synchronize(device)
    runtime = time.perf_counter()-start
    metrics.update({key:None for key in MCMC_FIELDS})
    metrics.update(runtime_sec=runtime, sampling_sec=runtime, final_render_sec=0.,
                   nominal_samples=total, samples_per_sec=total/runtime,
                   diagnostics_sec=0., cold_acceptance=None, minimum_swap_acceptance=None,
                   denoiser_evaluations=total*sampler.num_timesteps,
                   guidance_gradient_evaluations=total*sampler.num_timesteps)
    traces = {"log_likelihood":-arrays["phi"], "image_summaries":arrays["image_summaries"]}
    return metrics, arrays, traces, {"note":"Independent DPS trajectories; MCMC ESS and R-hat are not applicable."}


def run(config, manifest, resume=False, preflight_only=False):
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run on a GPU host; CPU validation uses tests instead of full ImageNet models.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    outdir = Path(config["outdir"]).resolve()
    outdir.mkdir(parents=True,exist_ok=True)
    provenance = {"images": {r["image_id"]:file_hash(r["image_path"]) for r in manifest},
                  "code":{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(ROOT.rglob("*.py"))},
                  "checkpoints":{}}
    families = []
    if any(m.startswith("imf_") for m in config["methods"]):
        families.append("imf")
    if "diffusion_dps" in config["methods"]:
        families.append("dps")
    for family in families:
        provenance["checkpoints"][family] = file_hash(config[family]["checkpoint"])
    contract = {"config":config,"manifest":manifest,"provenance":provenance,
                "preprocessing":"EXIF orientation, RGB, resize shortest side with bicubic, center crop 256, scale to [-1,1]",
                "pixel_likelihood":"Gaussian sum of squared residuals / (2 sigma^2); no pixel clipping",
                "conditioning":"Both priors receive the same known ImageNet class ID; iMF also uses its fixed CFG settings",
                "trace_axes":"iMF: draw, independent cold ladder, coordinate; DPS: independent output, summary"}
    protocol_path = outdir/"protocol.json"
    if protocol_path.exists():
        if not resume:
            raise FileExistsError(f"{outdir} already contains a run. Use --resume for the same protocol or choose a new directory.")
        previous = json.loads(protocol_path.read_text())
        if previous["contract"] != contract:
            raise ValueError("Resume protocol, image, code, or checkpoint mismatch; choose a new output directory")
    else:
        write_json(protocol_path,{"contract":contract,"environment":{
            "python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,
            "device":str(device),"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None}})
    jobs = prepare_jobs(config,manifest,outdir,resume)
    for family in families:
        methods = [m for m in config["methods"] if (m.startswith("imf_") if family=="imf" else m=="diffusion_dps")]
        load_start = time.perf_counter()
        if family == "imf":
            transport, decoder = load_imf(config["imf"],device)
            log(f"iMF transport: steps={config['imf']['steps']}, "
                f"guidance_interval={config['imf']['guidance_interval']}, "
                f"activation_checkpointing={config['imf']['activation_checkpointing']}")
        else:
            model, sampler = load_dps(config["dps"],device)
        synchronize(device)
        load_sec = time.perf_counter()-load_start
        checked = set()
        for job in jobs:
            likelihood = make_likelihood(job,config,device)
            if family == "imf":
                problem = ImageProblem(transport,decoder,likelihood,job["class_id"],device,
                                       config["likelihood_batch_size"])
                problem.truth = job["truth"].to(device)
                if job["task"] not in checked:
                    log(f"Preflight: iMF + decoder + {job['task']} likelihood")
                    info = preflight(problem, finite_difference_check=config.get("finite_difference_check", False))
                    write_json(outdir/f"preflight_{job['task']}.json",info)
                    checked.add(job["task"])
            if preflight_only:
                if family == "dps" and job["task"] not in checked:
                    # Actual U-Net backward and native DDPM update, not a fake weight fallback.
                    from .dps import BatchPosteriorSampling
                    probe = torch.zeros(1,3,256,256,device=device,requires_grad=True)
                    t = torch.full((1,),sampler.num_timesteps-1,device=device,dtype=torch.long)
                    kwargs = ({"y":torch.full((1,),job["class_id"],device=device,dtype=torch.long)}
                              if config["dps"]["class_cond"] else {})
                    out = sampler.p_sample(model=model,x=probe,t=t,model_kwargs=kwargs)
                    condition = BatchPosteriorSampling(likelihood.operator,None,
                                      scale=config["dps"]["guidance_scale"][job["task"]])
                    gradient, norm = condition.grad_and_value(probe,out["pred_xstart"],likelihood.measurement)
                    if not torch.isfinite(gradient).all():
                        raise RuntimeError("DPS preflight gradient is nonfinite")
                    write_json(outdir/f"preflight_dps_{job['task']}.json",{"residual_norm":float(norm),"gradient_norm":float(gradient.norm())})
                    del probe,out,gradient
                    checked.add(job["task"])
                continue
            for seed in config["sampler_seeds"]:
                for method in methods:
                    destination = outdir/"runs"/job["task"]/job["image_id"]/method/f"seed_{seed}"
                    if resume and (destination/"result.json").is_file():
                        log(f"Skipping completed {job['image_id']} / {job['task']} / {method} / {seed}")
                        continue
                    destination.mkdir(parents=True,exist_ok=True)
                    log(f"Running {job['image_id']} / {job['task']} / {method} / seed {seed}")
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    if family == "imf":
                        metrics,arrays,traces,details = run_imf(problem,config,method,seed)
                    else:
                        metrics,arrays,traces,details = run_dps(model,sampler,job,likelihood,config,seed,device)
                    metrics = {"task":job["task"],"image_id":job["image_id"],"class_id":job["class_id"],
                               "method":method,"seed":seed,**metrics,"model_load_sec":load_sec,
                               "peak_cuda_allocated_mb":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else None}
                    np.savez_compressed(destination/"traces.npz",**traces)
                    np.savez_compressed(destination/"posterior_summary.npz",mean=arrays["mean"],std=arrays["std"],
                                        examples=arrays["examples"],sample_psnr=arrays["psnr"])
                    save_images(destination,job["truth"][0].numpy(),job["measurement"][0].numpy(),job["task"],arrays)
                    write_json(destination/"result.json",{"metrics":metrics,"details":details})
                    refresh_report(outdir)
                    log(f"Saved: time={metrics['runtime_sec']:.1f}s; latent ESS/s={metrics.get('latent_coordinate_ess_mean_per_sec')}")
                    del arrays,traces,details
        if family == "imf":
            del problem,transport,decoder
        else:
            del model,sampler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    refresh_report(outdir)
