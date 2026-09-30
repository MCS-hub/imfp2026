"""Native DPS reverse diffusion, with independent per-image guidance norms."""
from pathlib import Path
import torch
import yaml
from .vendor import ROOT, enable_vendor
from .models import freeze

enable_vendor()
from guided_diffusion.condition_methods import PosteriorSampling


class BatchPosteriorSampling(PosteriorSampling):
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        residual = self.operator(x_0_hat) - measurement
        # Upstream sums all images in ONE norm. This extension equals separate
        # batch-size-one DPS updates and prevents a sqrt(batch_size) scale change.
        norms = torch.linalg.vector_norm(residual.flatten(1), dim=1)
        gradient = torch.autograd.grad(norms.sum(), x_prev)[0]
        return gradient, norms.detach()


def load_dps(config, device):
    from guided_diffusion.unet import create_model
    from guided_diffusion.gaussian_diffusion import create_sampler
    if not Path(config["checkpoint"]).is_file():
        raise FileNotFoundError(f"DPS checkpoint missing: {config['checkpoint']}")
    root = ROOT / "vendor" / "diffusion-posterior-sampling-main" / "configs"
    with (root / "imagenet_model_config.yaml").open() as f:
        architecture = yaml.safe_load(f)
    architecture["model_path"] = str(config["checkpoint"])
    # Override the archived DPS default: OpenAI's 256x256_diffusion.pt is
    # class-conditional. Strict loading rejects an incompatible checkpoint.
    architecture["class_cond"] = bool(config["class_cond"])
    model = freeze(create_model(**architecture)).to(device)
    with (root / "diffusion_config.yaml").open() as f:
        diffusion = yaml.safe_load(f)
    diffusion["timestep_respacing"] = str(config["steps"])
    sampler = create_sampler(**diffusion)
    return model, sampler


def sample_dps(model, sampler, likelihood, n_samples, batch_size, seed, image_shape,
               device, guidance_scale, class_id, class_cond, consume, progress=None):
    """Call consume(images) for final independent trajectories; no MCMC ESS."""
    device = torch.device(device)
    conditioner = BatchPosteriorSampling(likelihood.operator, None, scale=guidance_scale)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # The supplied DDPM implementation draws from torch's default generator.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        for first in range(0, n_samples, batch_size):
            count = min(batch_size, n_samples-first)
            x = torch.randn(count, *image_shape, device=device)
            model_kwargs = ({"y": torch.full((count,), int(class_id), device=device,
                                             dtype=torch.long)} if class_cond else {})
            for step in reversed(range(sampler.num_timesteps)):
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    t = torch.full((count,), step, device=device, dtype=torch.long)
                    out = sampler.p_sample(model=model, x=x, t=t, model_kwargs=model_kwargs)
                    updated, distance = conditioner.conditioning(
                        x_prev=x, x_t=out["sample"], x_0_hat=out["pred_xstart"],
                        measurement=likelihood.measurement)
                x = updated.detach()
                if not torch.isfinite(x).all():
                    raise FloatingPointError(f"DPS diverged at reverse step {step}; tune guidance on a pilot set")
            consume(x)
            if progress:
                progress(f"DPS outputs {first+count}/{n_samples}; residual norm={float(distance.mean()):.3g}")
