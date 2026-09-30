"""Synthetic-compatible ESS, split R-hat, and streamed image summaries."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def coordinate_ess(trace, chunk=64):
    """Sum initial-positive-sequence ESS across separate ladders.

    Input axes are [draw, ladder, coordinate]. Rejected repeats are retained.
    This reproduces imfp_scripts/evaluation.py's estimator and N cap, using
    batched FFTs. It is not ArviZ bulk/tail ESS. Stuck chains contribute zero.
    """
    a = np.asarray(trace)
    if a.ndim == 2:
        a = a[..., None]
    a = a.reshape(a.shape[0], a.shape[1], -1)
    n, chains, dim = a.shape
    if n < 4 or not np.isfinite(a).all():
        return np.full(dim, np.nan)
    result = np.zeros(dim)
    nfft = 1 << (2*n-1).bit_length()
    for start in range(0, dim, chunk):
        values = np.asarray(a[..., start:start+chunk], dtype=np.float64)
        values = values - values.mean(axis=0)
        spectrum = np.fft.rfft(values, n=nfft, axis=0)
        acov = np.fft.irfft(spectrum * spectrum.conj(), n=nfft, axis=0)[:n]
        acov /= np.arange(n, 0, -1)[:, None, None]
        valid = acov[0] > 1e-15
        rho = np.divide(acov, acov[0], out=np.zeros_like(acov), where=valid[None])
        tau = np.ones_like(acov[0])
        active = valid.copy()
        for lag in range(1, n-1, 2):
            pair = rho[lag] + rho[lag+1]
            active &= pair > 0
            tau += 2 * np.where(active, pair, 0)
        result[start:start+chunk] = np.where(valid, n / np.maximum(tau, 1), 0).sum(axis=0)
    return result


def split_rhat(trace):
    a = np.asarray(trace, dtype=np.float64)
    if a.ndim == 2:
        a = a[..., None]
    a = a.reshape(a.shape[0], a.shape[1], -1)
    half = len(a)//2
    if half < 4 or a.shape[1] < 2 or not np.isfinite(a).all():
        return np.full(a.shape[-1], np.nan)
    split = np.concatenate((a[:half], a[-half:]), axis=1)
    within = split.var(axis=0, ddof=1).mean(axis=0)
    between = half * split.mean(axis=0).var(axis=0, ddof=1)
    variance = (half-1)/half * within + between/half
    answer = np.sqrt(np.divide(variance, within, out=np.full_like(within, np.nan), where=within>0))
    answer[(within == 0) & (between > 0)] = np.inf
    return answer


def mcmc_diagnostics(latents, log_likelihood, runtime_sec, image_summaries=None):
    ess = coordinate_ess(latents)
    rhat = split_rhat(latents)
    ll_ess = coordinate_ess(log_likelihood)[0]
    row = {
        "latent_coordinate_ess_mean": float(np.mean(ess)),
        "latent_coordinate_ess_median": float(np.median(ess)),
        "latent_coordinate_ess_p05": float(np.quantile(ess, .05)),
        "latent_coordinate_ess_min": float(np.min(ess)),
        "latent_coordinate_ess_mean_per_sec": float(np.mean(ess)/runtime_sec),
        "log_likelihood_ess": float(ll_ess),
        "log_likelihood_ess_per_sec": float(ll_ess/runtime_sec),
        "latent_split_rhat_median": float(np.median(rhat)),
        "latent_split_rhat_max": float(np.max(rhat)),
        "log_likelihood_split_rhat": float(split_rhat(log_likelihood)[0]),
    }
    if image_summaries is not None:
        im_ess = coordinate_ess(image_summaries)
        row["image_summary_ess_mean"] = float(im_ess.mean())
        row["image_summary_ess_mean_per_sec"] = float(im_ess.mean()/runtime_sec)
    return row, ess, rhat


MCMC_FIELDS = (
    "latent_coordinate_ess_mean", "latent_coordinate_ess_median", "latent_coordinate_ess_p05",
    "latent_coordinate_ess_min", "latent_coordinate_ess_mean_per_sec", "log_likelihood_ess",
    "log_likelihood_ess_per_sec", "latent_split_rhat_median", "latent_split_rhat_max",
    "log_likelihood_split_rhat", "image_summary_ess_mean", "image_summary_ess_mean_per_sec",
)


class ImageAccumulator:
    """Constant image-memory cost; only a few examples and small traces are kept."""
    def __init__(self, truth, likelihood, example_indices):
        self.truth = truth
        self.likelihood = likelihood
        self.indices = set(int(i) for i in example_indices)
        self.examples = []
        self.n = 0
        self.mean = torch.zeros_like(truth[0], dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)
        self.phi, self.psnr, self.image_summaries = [], [], []

    @torch.no_grad()
    def consume(self, images):
        if not torch.isfinite(images).all():
            raise FloatingPointError("Nonfinite reconstructed pixels")
        b = len(images)
        values = images.double()
        batch_mean = values.mean(0)
        batch_m2 = (values-batch_mean).square().sum(0)
        delta = batch_mean-self.mean
        self.m2 += batch_m2 + delta.square() * self.n*b/(self.n+b)
        self.mean += delta * b/(self.n+b)
        self.phi.append(self.likelihood(images).cpu().numpy())
        # Pixel range is [-1,1], so the PSNR data range is 2. No clipping.
        mse = (values-self.truth.double()).square().flatten(1).mean(1)
        self.psnr.append((10*torch.log10(4/mse.clamp_min(1e-30))).cpu().numpy())
        self.image_summaries.append(F.adaptive_avg_pool2d(images, (4,4)).flatten(1).cpu().numpy())
        for k in range(b):
            if self.n+k in self.indices:
                self.examples.append(images[k].cpu().numpy())
        self.n += b

    def finish(self):
        phi = np.concatenate(self.phi)
        psnr = np.concatenate(self.psnr)
        mean = self.mean.float().cpu().numpy()
        std = (self.m2 / max(self.n-1, 1)).clamp_min(0).sqrt().float().cpu().numpy()
        mse_mean = float((self.mean-self.truth[0].double()).square().mean())
        residual_rmse = np.sqrt(2*phi*self.likelihood.sigma**2/self.likelihood.measurement_dim)
        metrics = {"sample_psnr_mean": float(psnr.mean()), "sample_psnr_std": float(psnr.std(ddof=1)) if self.n>1 else 0.,
                   "posterior_mean_psnr": 10*math.log10(4/max(mse_mean, 1e-30)),
                   "measurement_rmse_mean": float(residual_rmse.mean()),
                   "mean_pixel_std": float(std.mean()), "log_likelihood_mean": float(-phi.mean())}
        return metrics, {"mean": mean, "std": std, "examples": np.stack(self.examples),
                         "phi": phi, "psnr": psnr,
                         "image_summaries": np.concatenate(self.image_summaries)}
