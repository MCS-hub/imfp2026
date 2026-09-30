"""Isolate a saved run's gradient mismatch without starting a sampler."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from image_benchmark.models import ImageProblem, load_imf
from image_benchmark.observations import PixelLikelihood, make_operator


def check(name, fn, x, gradient):
    direction = gradient / gradient.norm()
    analytic = (gradient.double() * direction.double()).sum().item()
    print(f"{name}: analytic={analytic:.9e}", flush=True)
    for h in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
        with torch.no_grad():
            fd = ((fn(x+h*direction).double()-fn(x-h*direction).double())/(2*h)).item()
        print(f"  h={h:.1e} fd={fd:.9e} relative={abs(fd-analytic)/max(abs(analytic),1e-12):.6g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.run_dir / "protocol.json").read_text())["contract"]
    config = contract["config"]
    item = contract["manifest"][0]
    task = config["tasks"][0]
    print(json.dumps(config["imf"], indent=2), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(config["device"])
    transport, decoder = load_imf(config["imf"], device)
    obs_path = args.run_dir / "observations" / task / item["image_id"] / "observation.npz"
    with np.load(obs_path) as obs:
        measurement = torch.from_numpy(obs["measurement"].copy()).to(device)
    likelihood = PixelLikelihood(make_operator(task, config["observations"][task]).to(device),
                                 measurement, config["observations"][task]["sigma"])
    problem = ImageProblem(transport, decoder, likelihood, item["class_id"], device)
    rng = torch.Generator(device=device).manual_seed(732)
    z = torch.randn(1, 4096, generator=rng, device=device)
    phi, full_gradient = problem.phi_and_grad(z)
    print(f"Full checkpointed phi={phi.item():.9e}", flush=True)
    check("full transport + decoder", problem.phi_torch, z, full_gradient)

    # Separate VJPs avoid holding all XL model graphs in memory simultaneously.
    labels = torch.full((1,), item["class_id"], device=device, dtype=torch.long)
    times = torch.linspace(1, 0, transport.steps+1, device=device)
    def step(x, i):
        b = len(x)
        h = times[i]-times[i+1]
        u = transport.model.u_fn(x, times[i].expand(b), h.expand(b),
            x.new_full((b,), transport.omega), x.new_full((b,), transport.interval[0]),
            x.new_full((b,), transport.interval[1]), labels)[0]
        return x-h*u
    states = [z.reshape(1, 4, 32, 32)]
    with torch.no_grad():
        for i in range(transport.steps):
            states.append(step(states[-1], i))
    decoder.activation_checkpointing = False
    x = states[-1].detach().requires_grad_(True)
    value = likelihood(decoder(x))
    gradient = torch.autograd.grad(value.sum(), x)[0].detach()
    check("decoder + likelihood (no checkpoint)", lambda a: likelihood(decoder(a)), x.detach(), gradient)
    del value, x
    for i in reversed(range(transport.steps)):
        upstream = gradient
        x = states[i].detach().requires_grad_(True)
        y = step(x, i)
        gradient = torch.autograd.grad(y, x, upstream)[0].detach()
        check(f"transport step {i+1} VJP (no checkpoint)",
              lambda a: (step(a, i).double()*upstream.double()).sum(), x.detach(), gradient)
        del x, y
    difference = (gradient.flatten(1)-full_gradient).double().norm()/full_gradient.double().norm()
    print(f"Checkpointed vs separate uncheckpointed VJPs relative L2={difference.item():.9e}", flush=True)


if __name__ == "__main__":
    main()
