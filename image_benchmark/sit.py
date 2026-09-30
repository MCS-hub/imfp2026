"""Deterministic linear-interpolant SiT transport for source-space pCN."""
import importlib.util
import torch
from torch import nn
from .models import freeze
from .vendor import ROOT


class SiTTransport(nn.Module):
    """Fixed-step Heun, noise t=0 to data t=1; two velocity calls per step.

    CFG reproduces the official three-channel convention and evolves both
    conditional/unconditional states, including their distinct fourth channels.
    Counters count batched calls and individual states evaluated (including CFG).
    """
    def __init__(self, model, steps=125, cfg_scale=4.0):
        super().__init__()
        self.model = freeze(model)
        self.steps, self.cfg_scale = steps, cfg_scale
        self.network_calls = self.network_state_evaluations = 0

    @torch.no_grad()
    def forward(self, source, class_id):
        x = source.reshape(-1, 4, 32, 32)
        batch = len(x)
        labels = torch.full((batch,), int(class_id), device=x.device, dtype=torch.long)
        if self.cfg_scale > 1:
            x = torch.cat([x, x], 0)
            labels = torch.cat([labels, torch.full_like(labels, 1000)])
        def velocity(state, time):
            t = state.new_full((len(state),), time)
            self.network_calls += 1
            self.network_state_evaluations += len(state)
            if self.cfg_scale > 1:
                return self.model.forward_with_cfg(state, t, labels, self.cfg_scale)
            return self.model(state, t, labels)
        dt = 1.0 / self.steps
        for i in range(self.steps):
            first = velocity(x, i * dt)
            second = velocity(x + dt * first, (i + 1) * dt)
            x = x + (dt / 2) * (first + second)
        return x[:batch]


class SiTDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = freeze(vae)

    def forward(self, latent):
        return self.vae.decode(latent / 0.18215).sample


def load_sit(config, device):
    from safetensors.torch import load_file
    from diffusers import AutoencoderKL
    spec = importlib.util.spec_from_file_location('benchmark_vendor_sit', ROOT/'vendor/sit/models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = load_file(config['checkpoint'], device='cpu')
    # Infer the released head layout; strict loading verifies every other shape.
    out_channels = state['final_layer.linear.weight'].shape[0] // 4
    if out_channels not in (4, 8):
        raise ValueError('Unexpected SiT output head')
    model = module.SiT_XL_2(input_size=32, num_classes=1000, learn_sigma=out_channels == 8)
    model.load_state_dict(state, strict=True)
    vae = AutoencoderKL.from_pretrained(config['vae'], torch_dtype=torch.float32,
                                       local_files_only=config['local_files_only'])
    del vae.encoder
    return (SiTTransport(model, config['steps'], config['cfg_scale']).to(device),
            SiTDecoder(vae).to(device))


@torch.no_grad()
def preflight_sit(problem):
    rng = torch.Generator(device=problem.device).manual_seed(732)
    z = torch.randn(1, problem.dim, generator=rng, device=problem.device, dtype=problem.dtype)
    value = problem.phi_torch(z)
    again = problem.phi_torch(z)
    latent, image = problem.render(z)
    if not all(torch.isfinite(v).all() for v in (value, again, latent, image)):
        raise RuntimeError('Nonfinite SiT preflight output')
    torch.testing.assert_close(value, again, rtol=0, atol=0)
    return {'phi':float(value), 'deterministic_repeat':True,
            'gradient_check':'not required: pCN uses values only',
            'latent_shape':list(latent.shape), 'pixel_shape':list(image.shape)}
