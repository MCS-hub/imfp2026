#!/usr/bin/env python3
"""Empirical loss-to-map-error experiment for Theorem 3.7 (standalone).

Requires Python >= 3.10, torch >= 2.0, numpy, scipy, matplotlib.
    python banana_map_error_2d.py --device cpu --quick --outdir outs/map_error_smoke
    python banana_map_error_2d.py --device cuda --outdir outs/map_error_2d
    python banana_map_error_2d.py --self-test
    python banana_map_error_2d.py --plot-only --outdir outs/banana_map_error_2d

Training defaults to one seed (0). Figures use one run and are saved separately;
--plot-seed selects a run when regenerating figures from older multi-seed results.

The reference is the marginal ODE flow for independent interpolation, NOT the
explicit triangular banana generator. All reference computations use float64.
Delta is the held-out marginal joint risk (sums over coordinates), not the noisy
samplewise training loss. Fixed independent evaluation draws are shared across
checkpoints/seeds. Error bars are Monte Carlo standard errors, conditional on a
trained model and the numerical reference. They exclude reference approximation
and training-seed uncertainty. This experiment does not certify a bound or an
optimal power law: regularity constants may change during training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import roots_hermitenorm, logsumexp
import torch
from torch import nn
from torch.func import jvp


def banana_samples(rng, count, curvature, tau):
    uv = rng.standard_normal((count, 2))
    return np.column_stack((uv[:, 0], curvature * (uv[:, 0] ** 2 - 1) + tau * uv[:, 1]))


class ReferenceVelocity:
    """Integrate U | W1, then reweight by p(W2 | U).

    X=(U, b(U^2-1)+tau*V), W=(1-t)X+tZ, independent U,V,Z standard
    normal. Given W1, U=m+s*e where e~N(0,1), m=(1-t)W1/D1,
    s=t/sqrt(D1), D1=(1-t)^2+t^2. W2|U has mean (1-t)f(U)
    and variance D2=(1-t)^2*tau^2+t^2. Gauss-Hermite quadrature
    computes the reweighted E[e] and E[f(U)]. The algebra below avoids
    dividing by t near zero and gives v(w,0)=-w and v(w,1)=w.
    """
    def __init__(self, curvature=0.62, tau=0.30, order=128, batch_size=512):
        self.curvature, self.tau = curvature, tau
        self.nodes, weights = roots_hermitenorm(order)
        # Extremely small quadrature weights can underflow at large orders.
        with np.errstate(divide='ignore'):
            self.log_weights = np.log(weights) - 0.5 * np.log(2 * np.pi)
        self.batch_size = batch_size
        self.evaluations = 0

    def __call__(self, w, t):
        w = np.asarray(w, dtype=np.float64)
        if w.ndim != 2 or w.shape[1] != 2:
            raise ValueError('Reference velocity expects (N,2) inputs')
        times = np.broadcast_to(np.asarray(t, dtype=np.float64).reshape(-1), (len(w),))
        result = np.empty_like(w)
        for lo in range(0, len(w), self.batch_size):
            hi = min(lo + self.batch_size, len(w))
            wt, tt = w[lo:hi], times[lo:hi, None]
            a = 1 - tt
            d1 = a * a + tt * tt
            d2 = a * a * self.tau ** 2 + tt * tt
            u = a * wt[:, :1] / d1 + tt / np.sqrt(d1) * self.nodes[None, :]
            f = self.curvature * (u * u - 1)
            logp = self.log_weights[None, :] - (wt[:, 1:2] - a * f) ** 2 / (2 * d2)
            probabilities = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
            mean_e = np.sum(probabilities * self.nodes, axis=1)
            mean_f = np.sum(probabilities * f, axis=1)
            result[lo:hi, 0] = ((tt - a) / d1).ravel() * wt[:, 0] - mean_e / np.sqrt(d1).ravel()
            result[lo:hi, 1] = (((tt - a * self.tau ** 2) * wt[:, 1:2] - tt * mean_f[:, None]) / d2).ravel()
        self.evaluations += len(w)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite reference velocity')
        return result


def reference_map(source, velocity, rtol=1e-8, atol=1e-10, batch_size=128):
    mapped = np.empty_like(source, dtype=np.float64)
    for lo in range(0, len(source), batch_size):
        z = source[lo:lo + batch_size]
        solution = solve_ivp(
            lambda t, flat: velocity(flat.reshape(-1, 2), t).ravel(),
            (1.0, 0.0), z.ravel(), method='DOP853', rtol=rtol, atol=atol,
        )
        if not solution.success:
            raise RuntimeError('Reference ODE failed: ' + solution.message)
        mapped[lo:lo + len(z)] = solution.y[:, -1].reshape(-1, 2)
    return mapped


class IMF(nn.Module):
    def __init__(self, width=128, depth=3):
        super().__init__()
        layers = [nn.Linear(4, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        self.backbone = nn.Sequential(*layers)
        self.head_u, self.head_v = nn.Linear(width, 2), nn.Linear(width, 2)

    def forward(self, w, r, t):
        return self.head_u(self.backbone(torch.cat((w, r, t), dim=1)))

    def velocity(self, w, t):
        return self.head_v(self.backbone(torch.cat((w, t, t), dim=1)))


def reconstructed_velocity(model, w, r, t, stop_gradient=False):
    v = model.velocity(w, t)
    u, derivative = jvp(model, (w, r, t),
                        (v.detach() if stop_gradient else v,
                         torch.zeros_like(r), torch.ones_like(t)))
    if stop_gradient:
        derivative = derivative.detach()
    return u + (t - r) * derivative, v


@torch.no_grad()
def learned_map(model, z, steps):
    w = z.clone()
    for k in range(steps, 0, -1):
        t, r = w.new_full((len(w), 1), k / steps), w.new_full((len(w), 1), (k - 1) / steps)
        w = w - (t - r) * model(w, r, t)
    return w


def mean_se(values):
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


@torch.no_grad()
def evaluate(model, evaluation, map_steps, auxiliary_weight, device, batch_size):
    components = {'imf': [], 'fm': [], 'raw': []}
    dtype = next(model.parameters()).dtype
    for lo in range(0, len(evaluation['w']), batch_size):
        sl = slice(lo, lo + batch_size)
        w, r, t, v, target = [torch.as_tensor(evaluation[key][sl], device=device, dtype=dtype)
                              for key in ['w', 'r', 't', 'v_reference', 'target']]
        predicted, auxiliary = reconstructed_velocity(model, w, r, t)
        components['imf'].append((predicted - v).square().sum(1).cpu().numpy())
        components['fm'].append((auxiliary - v).square().sum(1).cpu().numpy())
        components['raw'].append(((predicted - target).square().sum(1)
                                  + auxiliary_weight * (auxiliary - target).square().sum(1)).cpu().numpy())
    components = {key: np.concatenate(value).astype(np.float64) for key, value in components.items()}
    delta, delta_se = mean_se(components['imf'] + auxiliary_weight * components['fm'])
    raw, raw_se = mean_se(components['raw'])
    summary = dict(delta=delta, delta_mc_se=delta_se,
                   risk_imf=float(components['imf'].mean()), risk_fm=float(components['fm'].mean()),
                   raw_joint_loss=raw, raw_joint_loss_mc_se=raw_se)
    # Empirical spatial Jacobian diagnostic; not a certified global Lipschitz bound.
    n = min(256, len(evaluation['w']))
    w, r, t = [torch.as_tensor(evaluation[key][:n], device=device, dtype=dtype) for key in ['w', 'r', 't']]
    columns = []
    for coordinate in range(2):
        direction = torch.zeros_like(w)
        direction[:, coordinate] = 1
        _, column = jvp(model, (w, r, t), (direction, torch.zeros_like(r), torch.zeros_like(t)))
        columns.append(column)
    norms = torch.linalg.svdvals(torch.stack(columns, dim=-1))[:, 0]
    summary['sampled_spatial_jacobian_max'] = float(norms.max())
    summary['sampled_spatial_jacobian_mean'] = float(norms.mean())
    rows = []
    for steps in map_steps:
        errors = []
        for lo in range(0, len(evaluation['source']), batch_size):
            sl = slice(lo, lo + batch_size)
            z = torch.as_tensor(evaluation['source'][sl], device=device, dtype=dtype)
            prediction = learned_map(model, z, steps).cpu().numpy().astype(np.float64)
            errors.append(np.sum((prediction - evaluation['reference_map'][sl]) ** 2, axis=1))
        error, error_se = mean_se(np.concatenate(errors))
        rows.append(dict(summary, map_steps=steps, map_mse=error, map_mse_mc_se=error_se,
                         map_l2=math.sqrt(error)))
    if not all(np.isfinite(value) for row in rows for value in row.values()):
        raise FloatingPointError('Nonfinite checkpoint metrics')
    return rows


def build_reference(args):
    rng = np.random.default_rng(args.eval_seed)
    source = rng.standard_normal((args.map_samples, 2))
    x = banana_samples(rng, args.risk_samples, args.curvature, args.tau)
    z = rng.standard_normal(x.shape)
    r, t = rng.random((args.risk_samples, 1)), rng.random((args.risk_samples, 1))
    w = (1 - t) * x + t * z
    reference = ReferenceVelocity(args.curvature, args.tau, args.quadrature_order)
    refined = ReferenceVelocity(args.curvature, args.tau, 2 * args.quadrature_order)
    print('Computing reference velocity and marginal ODE map...', flush=True)
    started = time.perf_counter()
    v = reference(w, t.ravel())
    v_refined = refined(w, t.ravel())
    mapped = reference_map(source, reference, args.ode_rtol, args.ode_atol)
    n = min(args.validation_samples, len(source))
    # Separate quadrature sensitivity from ODE integration sensitivity.
    mapped_quadrature = reference_map(source[:n], refined, args.ode_rtol, args.ode_atol)
    mapped_tight = reference_map(source[:n], reference, args.ode_rtol / 10, args.ode_atol / 10)
    rms = lambda a: float(np.sqrt(np.mean(np.sum(a * a, axis=1))))
    velocity_gap = rms(v - v_refined)
    map_gap, ode_gap = rms(mapped[:n] - mapped_quadrature), rms(mapped[:n] - mapped_tight)
    diagnostics = dict(quadrature_order=args.quadrature_order, validation_order=2 * args.quadrature_order,
                       velocity_quadrature_rms=velocity_gap, map_quadrature_rms=map_gap,
                       map_ode_rms=ode_gap, validation_map_samples=n,
                       map_reference_sensitivity_mse=max(map_gap, ode_gap) ** 2,
                       reference_seconds=time.perf_counter() - started,
                       reference_mean=mapped.mean(axis=0).tolist(),
                       reference_covariance=np.cov(mapped, rowvar=False).tolist(),
                       analytic_prior_mean=[0.0, 0.0],
                       analytic_prior_covariance=[[1.0, 0.0], [0.0, 2 * args.curvature ** 2 + args.tau ** 2]])
    diagnostics['accuracy_check_passed'] = max(velocity_gap, map_gap, ode_gap) <= args.reference_tolerance
    print(json.dumps(diagnostics, indent=2), flush=True)
    with (args.outdir / 'reference_checks.json').open('w') as f:
        json.dump(diagnostics, f, indent=2)
    if not diagnostics['accuracy_check_passed']:
        raise RuntimeError('Reference sensitivity exceeds --reference-tolerance. Increase '
                           '--quadrature-order and/or tighten --ode-rtol/--ode-atol; use a new output directory.')
    data = dict(w=w, r=r, t=t, target=z - x, v_reference=v,
                source=source, reference_map=mapped)
    np.savez_compressed(args.outdir / 'reference.npz', **data)
    return data, diagnostics


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(outdir, rows, diagnostics, plot_seed=None, font_size=16):
    """Export three separate figures for one run, keeping all saved metrics intact."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    available = sorted(set(int(row['seed']) for row in rows))
    if not available:
        raise ValueError('No metrics to plot')
    selected = available[0] if plot_seed is None else plot_seed
    if selected not in available:
        raise ValueError(f'Requested plotting seed {selected} not in {available}')
    selected_rows = sorted((row for row in rows if row['seed'] == selected),
                           key=lambda row: row['update'])
    steps_values = sorted(set(int(row['map_steps']) for row in selected_rows))
    style = {'font.size': font_size, 'axes.labelsize': font_size + 2,
             'axes.titlesize': font_size + 1, 'xtick.labelsize': font_size,
             'ytick.labelsize': font_size, 'legend.fontsize': font_size - 1,
             'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(style):
        def save(fig, ax, name):
            ax.grid(True, which='both', alpha=0.18)
            fig.tight_layout()
            for extension in ('png', 'pdf'):
                fig.savefig(outdir / f'{name}.{extension}', dpi=250, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 6))
        for index, steps in enumerate(steps_values):
            data = [row for row in selected_rows if row['map_steps'] == steps]
            delta = [row['delta'] for row in data]
            error = [row['map_mse'] for row in data]
            color = plt.get_cmap('tab10')(index)
            ax.errorbar(delta, error,
                        xerr=[row['delta_mc_se'] for row in data],
                        yerr=[row['map_mse_mc_se'] for row in data],
                        color=color, marker='o', markersize=4, linewidth=1.5,
                        elinewidth=0.8, alpha=0.85, label=f'$K={steps}$')
            ax.scatter(delta[-1], error[-1], color=color, marker='*', s=170, zorder=5)
        ax.set(xscale='log', yscale='log', xlabel=r'Marginal joint risk $\widehat\delta$',
               ylabel=r'Squared map error $\widehat E$')
        ax.legend(title='Transport steps', frameon=False)
        save(fig, ax, 'risk_vs_map_error')

        data = [row for row in selected_rows if row['map_steps'] == steps_values[0]]
        updates = [row['update'] for row in data]
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.plot(updates, [row['delta'] for row in data], linewidth=2,
                label='Marginal joint risk')
        ax.plot(updates, [row['raw_joint_loss'] for row in data], '--', linewidth=2,
                label='Raw joint loss')
        ax.set(yscale='log', xlabel='Optimizer updates', ylabel='Held-out squared-norm loss')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.legend(frameon=False)
        save(fig, ax, 'risk_vs_training')

        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.plot(updates, [row['sampled_spatial_jacobian_max'] for row in data], linewidth=2)
        ax.set(xlabel='Optimizer updates', ylabel='Sampled maximum Jacobian norm')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        save(fig, ax, 'spatial_jacobian_vs_training')
    with (outdir / 'plot_config.json').open('w') as f:
        json.dump({'selected_seed': selected, 'font_size': font_size,
                   'map_error': 'squared L2 error; stars mark final checkpoints',
                   'error_bars': 'conditional Monte Carlo standard errors',
                   'jacobian': 'sampled diagnostic, not a global Lipschitz bound',
                   'reference_sensitivity_mse': diagnostics['map_reference_sensitivity_mse']},
                  f, indent=2)
    print(f'Exported three separate figures using seed {selected} to {outdir}', flush=True)


def self_test():
    # Zero curvature gives a diagonal Gaussian prior with a known exact flow.
    rng = np.random.default_rng(101)
    w = rng.standard_normal((20, 2))
    t = np.linspace(0, 1, len(w))[:, None]
    tau = 0.3
    ref = ReferenceVelocity(0.0, tau, 64)
    expected = w * (t - (1 - t) * np.array([1.0, tau ** 2])) / (
        t ** 2 + (1 - t) ** 2 * np.array([1.0, tau ** 2]))
    np.testing.assert_allclose(ref(w, t.ravel()), expected, atol=1e-11)
    mapped = reference_map(w, ref, 1e-10, 1e-12)
    np.testing.assert_allclose(mapped, w * [1, tau], atol=2e-8)
    # Independent adaptive integration validates non-Gaussian conditional expectations.
    ref = ReferenceVelocity(0.62, tau, 128)
    for point, tt in [([0.8, -0.5], 0.2), ([-1.2, 1.0], 0.5), ([2.0, 0.8], 0.8)]:
        a = 1 - tt
        d1, d2 = a*a + tt*tt, a*a*tau*tau + tt*tt
        m, s = a * point[0] / d1, tt / np.sqrt(d1)
        fun = lambda e: 0.62 * ((m + s*e)**2 - 1)
        weight = lambda e: np.exp(-0.5*e*e - (point[1] - a*fun(e))**2 / (2*d2))
        norm = quad(weight, -12, 12, epsabs=1e-12)[0]
        ee = quad(lambda e: e*weight(e), -12, 12, epsabs=1e-12)[0] / norm
        ef = quad(lambda e: fun(e)*weight(e), -12, 12, epsabs=1e-12)[0] / norm
        expected = [(tt-a)/d1*point[0] - ee/np.sqrt(d1),
                    ((tt-a*tau*tau)*point[1]-tt*ef)/d2]
        np.testing.assert_allclose(ref(np.array([point]), tt)[0], expected, atol=2e-7)
    np.testing.assert_allclose(ref(w, 0), -w, atol=1e-12)
    np.testing.assert_allclose(ref(w, 1), w, atol=1e-12)
    # Check JVP against a central directional difference and exercise stop-gradient training.
    torch.manual_seed(4)
    model = IMF(16, 2).double()
    wt = torch.randn(8, 2, dtype=torch.float64)
    r, t = torch.rand(8, 1, dtype=torch.float64), torch.rand(8, 1, dtype=torch.float64)
    rec, velocity = reconstructed_velocity(model, wt, r, t)
    eps = 1e-5
    derivative_fd = (model(wt+eps*velocity, r, t+eps) - model(wt-eps*velocity, r, t-eps)) / (2*eps)
    torch.testing.assert_close(rec, model(wt, r, t)+(t-r)*derivative_fd, atol=1e-9, rtol=1e-7)
    rec, velocity = reconstructed_velocity(model, wt, r, t, True)
    (rec.square().sum() + velocity.square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('PASS: analytic Gaussian velocity/flow, independent banana quadrature, endpoints, JVP, training gradients.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', type=Path, default=Path('outs/banana_map_error_2d'))
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--updates', type=int, default=16000)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--eval-batch-size', type=int, default=1024)
    parser.add_argument('--training-samples', type=int, default=20000)
    parser.add_argument('--training-data-seed', type=int, default=1701)
    parser.add_argument('--eval-seed', type=int, default=9101)
    parser.add_argument('--map-samples', type=int, default=2048)
    parser.add_argument('--risk-samples', type=int, default=8192)
    parser.add_argument('--map-steps', type=int, nargs='+', default=[1, 6])
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--auxiliary-weight', type=float, default=1.0)
    parser.add_argument('--curvature', type=float, default=0.62)
    parser.add_argument('--tau', type=float, default=0.30)
    parser.add_argument('--quadrature-order', type=int, default=128)
    parser.add_argument('--validation-samples', type=int, default=128)
    parser.add_argument('--reference-tolerance', type=float, default=1e-3)
    parser.add_argument('--ode-rtol', type=float, default=1e-8)
    parser.add_argument('--ode-atol', type=float, default=1e-10)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--save-checkpoints', action='store_true')
    parser.add_argument('--plot-only', action='store_true', help='Regenerate figures from existing metrics without retraining')
    parser.add_argument('--plot-seed', type=int, default=None, help='Run to plot; defaults to the smallest saved training seed')
    parser.add_argument('--font-size', type=float, default=16, help='Base plot font size in points')
    parser.add_argument('--quick', action='store_true', help='Small CPU-compatible execution smoke test, not scientific evidence')
    parser.add_argument('--self-test', action='store_true', help='Validate numerical reference and training algebra, then exit')
    args = parser.parse_args()
    if not math.isfinite(args.font_size) or args.font_size < 8:
        parser.error('--font-size must be finite and at least 8')
    if args.plot_only:
        if not (args.outdir / 'metrics.csv').is_file() or not (args.outdir / 'reference_checks.json').is_file():
            parser.error('--plot-only requires metrics.csv and reference_checks.json in --outdir')
        with (args.outdir / 'metrics.csv').open() as f:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
        with (args.outdir / 'reference_checks.json').open() as f:
            diagnostics = json.load(f)
        plot_results(args.outdir, rows, diagnostics, args.plot_seed, args.font_size)
        return
    if args.self_test:
        torch.set_num_threads(1)
        self_test()
        return
    if args.quick:
        args.seeds, args.updates, args.eval_every = [0], 100, 50
        args.width, args.depth = 32, 2
        args.training_samples, args.batch_size = 1024, 128
        args.map_samples, args.risk_samples, args.validation_samples = 64, 256, 32
    for name in ['updates', 'eval_every', 'width', 'depth', 'batch_size', 'eval_batch_size',
                 'training_samples', 'quadrature_order', 'threads']:
        if getattr(args, name) < 1:
            parser.error(name + ' must be positive')
    if min(args.map_samples, args.risk_samples, args.validation_samples) < 2:
        parser.error('Sample counts must be at least two')
    if min(args.map_steps) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Map steps must be positive and seeds distinct')
    if min(args.tau, args.auxiliary_weight, args.learning_rate, args.reference_tolerance,
           args.ode_rtol, args.ode_atol) <= 0 or args.weight_decay < 0:
        parser.error('Scales/tolerances must be positive and weight decay nonnegative')
    if args.plot_seed is not None and args.plot_seed not in args.seeds:
        parser.error('--plot-seed must be among --seeds')
    if args.training_data_seed == args.eval_seed:
        parser.error('Training-data and evaluation seeds must differ')
    if args.outdir.exists() and any(args.outdir.iterdir()):
        parser.error('Output directory is nonempty; choose a new --outdir to preserve existing results')
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA/ROCm is unavailable; use --device cpu or run on a GPU node')
    config = dict(vars(args), outdir=str(args.outdir), actual_device=str(device),
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  status='running', loss_convention='risk metrics: squared Euclidean sums; training: coordinate averages',
                  reference='independent-interpolation marginal ODE, float64 quadrature + DOP853')
    with (args.outdir / 'config.json').open('w') as f:
        json.dump(config, f, indent=2)
    evaluation, diagnostics = build_reference(args)
    data = torch.as_tensor(banana_samples(np.random.default_rng(args.training_data_seed),
                           args.training_samples, args.curvature, args.tau), device=device, dtype=torch.float32)
    all_rows = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        model = IMF(args.width, args.depth).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                     betas=(0.9, 0.999), weight_decay=args.weight_decay)
        cursor, permutation, examples = len(data), None, 0
        for update in range(args.updates + 1):
            if update == 0 or update % args.eval_every == 0 or update == args.updates:
                model.eval()
                metrics = evaluate(model, evaluation, args.map_steps, args.auxiliary_weight, device, args.eval_batch_size)
                for metric in metrics:
                    all_rows.append(dict(seed=seed, update=update, examples_seen=examples, **metric))
                save_csv(args.outdir / 'metrics.csv', all_rows)
                print(f'seed={seed} update={update} delta={metrics[0]["delta"]:.6g} ' +
                      ' '.join(f'E(K={m["map_steps"]})={m["map_mse"]:.6g}' for m in metrics), flush=True)
                if args.save_checkpoints or update == args.updates:
                    torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                    seed=seed, update=update, config=config),
                               args.outdir / f'imf_seed{seed}_update{update}.pt')
            if update == args.updates:
                break
            model.train()
            if cursor >= len(data):
                permutation, cursor = torch.randperm(len(data), device=device), 0
            x = data[permutation[cursor:cursor + args.batch_size]]
            cursor += args.batch_size
            examples += len(x)
            t, r = torch.rand(len(x), 1, device=device), torch.rand(len(x), 1, device=device)
            z = torch.randn_like(x)
            w, target = (1-t)*x+t*z, z-x
            predicted, auxiliary = reconstructed_velocity(model, w, r, t, True)
            # Coordinate means match the existing synthetic training implementation.
            loss = (predicted-target).square().mean() + args.auxiliary_weight*(auxiliary-target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    plot_results(args.outdir, all_rows, diagnostics, args.plot_seed, args.font_size)
    config['status'] = 'complete'
    with (args.outdir / 'config.json').open('w') as f:
        json.dump(config, f, indent=2)
    print(f'Finished. Metrics, reference checks, models, and plots: {args.outdir}', flush=True)


if __name__ == '__main__':
    main()
