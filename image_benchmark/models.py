"""Deterministic, differentiable adapters for the released checkpoints."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .vendor import enable_vendor

LATENT_MEAN = (0.86488, -0.27787343, 0.21616915, 0.3738409)
LATENT_STD = (4.85503674, 5.31922414, 3.93725398, 3.9870003)


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


class IMFTransport(nn.Module):
    """Accept a source state, rather than drawing new noise as generate() does."""
    def __init__(self, model, steps=1, omega=8.0, interval=(0.4, 0.65),
                 activation_checkpointing=False):
        super().__init__()
        if steps < 1 or omega <= 0:
            raise ValueError("steps and omega must be positive")
        self.model = freeze(model)
        self.steps, self.omega = int(steps), float(omega)
        self.interval = tuple(interval)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, source, class_id):
        x = source.reshape(-1, 4, 32, 32)
        labels = torch.full((len(x),), int(class_id), device=x.device, dtype=torch.long)
        times = torch.linspace(1, 0, self.steps + 1, device=x.device, dtype=x.dtype)
        for i in range(self.steps):
            # Defaults capture each step for checkpoint recomputation during backward.
            def step(state, t=times[i], h=times[i] - times[i+1]):
                b = len(state)
                omega = state.new_full((b,), self.omega)
                lo = state.new_full((b,), self.interval[0])
                hi = state.new_full((b,), self.interval[1])
                u = self.model.u_fn(state, t.expand(b), h.expand(b), omega, lo, hi,
                                    y=labels)[0]
                return state - h * u
            if self.activation_checkpointing and torch.is_grad_enabled() and x.requires_grad:
                x = checkpoint(step, x, use_reentrant=False)
            else:
                x = step(x)
        return x


class LatentDecoder(nn.Module):
    def __init__(self, vae, activation_checkpointing=False):
        super().__init__()
        self.vae = freeze(vae)
        self.register_buffer("mean", torch.tensor(LATENT_MEAN).view(1, 4, 1, 1))
        self.register_buffer("std", torch.tensor(LATENT_STD).view(1, 4, 1, 1))
        self.activation_checkpointing = activation_checkpointing

    def forward(self, latent):
        # These checkpoint-specific statistics replace the usual SD scaling convention.
        scaled = latent * self.std + self.mean
        def decode(x):
            return self.vae.decode(x).sample
        if self.activation_checkpointing and torch.is_grad_enabled() and scaled.requires_grad:
            return checkpoint(decode, scaled, use_reentrant=False)
        return decode(scaled)


def load_imf(config, device):
    enable_vendor()
    from imf import iMeanFlow
    from diffusers import AutoencoderKL
    ckpt = Path(config["checkpoint"])
    if not ckpt.is_file():
        raise FileNotFoundError(f"iMF checkpoint missing: {ckpt}")
    model = iMeanFlow(config["architecture"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    transport = IMFTransport(model, config["steps"], config["omega"],
                             config["guidance_interval"], config["activation_checkpointing"])
    vae = AutoencoderKL.from_pretrained(config["vae"], torch_dtype=torch.float32,
                                       local_files_only=config["local_files_only"])
    del vae.encoder
    decoder = LatentDecoder(vae, config["activation_checkpointing"])
    return transport.to(device), decoder.to(device)


class ImageProblem:
    """Interface consumed by the supplied TorchSPT kernels. No decoder Jacobian needed."""
    def __init__(self, transport, decoder, likelihood, class_id, device, batch_size=1,
                 dim=4096, dtype=torch.float32):
        self.transport, self.decoder = transport, decoder
        self.likelihood, self.class_id = likelihood, int(class_id)
        self.device, self.dtype = torch.device(device), dtype
        self.batch_size, self.dim = int(batch_size), int(dim)
        self.scenario = "image"

    def phi_torch(self, source):
        leading = source.shape[:-1]
        flat = source.reshape(-1, self.dim)
        values = []
        for batch in flat.split(self.batch_size):
            values.append(self.likelihood(self.decoder(self.transport(batch, self.class_id))))
        return torch.cat(values).reshape(leading)

    def phi_and_grad(self, source):
        leading = source.shape[:-1]
        flat = source.detach().reshape(-1, self.dim)
        values, gradients = [], []
        # Finish backward separately for each microbatch; retain no large batch graph.
        with torch.enable_grad():
            for batch in flat.split(self.batch_size):
                batch = batch.detach().requires_grad_(True)
                value = self.likelihood(self.decoder(self.transport(batch, self.class_id)))
                gradient = torch.autograd.grad(value.sum(), batch)[0]
                values.append(value.detach())
                gradients.append(gradient.detach())
        return torch.cat(values).reshape(leading), torch.cat(gradients).reshape(source.shape)

    @torch.no_grad()
    def render(self, source):
        latent = self.transport(source, self.class_id)
        return latent, self.decoder(latent)
