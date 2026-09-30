import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from image_benchmark.models import IMFTransport
from image_benchmark.config import DEFAULT
from image_benchmark.vendor import enable_vendor

enable_vendor()
torch.set_num_threads(1)


class TinyTransport(nn.Module):
    def forward(self,z,class_id):
        return (.5*z).reshape(-1,4,32,32)


class TinyDecoder(nn.Module):
    def forward(self,x):
        return torch.nn.functional.interpolate(x[:,:3].tanh(),size=(256,256),mode="bilinear",align_corners=False)


class TinyDenoiser(nn.Module):
    def forward(self,x,t,y=None):
        if y is None:
            raise ValueError("class labels must reach the conditional denoiser")
        return torch.cat([.01*x,torch.zeros_like(x)],1)


class Integration(unittest.TestCase):
    def test_upstream_imf_forward_and_checkpointed_gradient(self):
        from imf import iMeanFlow
        from models import imfDiT
        def tiny(**kwargs):
            return imfDiT.imfDiT(hidden_size=32,depth=2,num_heads=4,aux_head_depth=1,**kwargs)
        with patch.object(imfDiT,"imfDiT_test",tiny,create=True):
            upstream=iMeanFlow("imfDiT_test")
        torch.manual_seed(1)
        with torch.no_grad():
            upstream.net.u_final_layer.linear._flax_linear.weight.normal_(0,.01)
            # Exercise attention/RoPE and MLP derivatives, which zero gates bypass.
            for block in list(upstream.net.shared_blocks) + list(upstream.net.u_heads):
                block.attn_scale.fill_(0.2)
                block.mlp_scale.fill_(0.2)
        source=torch.randn(2,4096)
        class FixedRNG:
            def randn(self,shape):
                return source.clone().reshape(shape)
        for steps in [1,2,4]:
            reference=upstream.generate(2,FixedRNG(),steps,8.,.4,.65,labels=torch.tensor([207,207]))
            transport=IMFTransport(upstream,steps,8.,(.4,.65))
            z=source.clone().requires_grad_(True)
            actual=transport(z,207)
            torch.testing.assert_close(actual,reference)
            gradient=torch.autograd.grad(actual.square().sum(),z)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()),0)
            direction = gradient / gradient.norm()
            with torch.no_grad():
                plus = transport(source + 1e-2*direction,207).double().square().sum()
                minus = transport(source - 1e-2*direction,207).double().square().sum()
            finite_difference = (plus-minus)/2e-2
            analytic = (gradient.double()*direction.double()).sum()
            torch.testing.assert_close(finite_difference,analytic,rtol=2e-3,atol=1e-3)
            transport.activation_checkpointing=True
            z2=source.clone().requires_grad_(True)
            checkpointed=transport(z2,207)
            checkpointed_gradient=torch.autograd.grad(checkpointed.square().sum(),z2)[0]
            torch.testing.assert_close(checkpointed,reference)
            torch.testing.assert_close(checkpointed_gradient,gradient)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in upstream.parameters()))

    def test_preflight_finite_differences_are_opt_in(self):
        from types import SimpleNamespace
        from image_benchmark.runner import preflight
        problem = SimpleNamespace(
            device="cpu", dim=2, dtype=torch.float32,
            phi_and_grad=lambda z: (z.square().sum(-1), 2*z),
            phi_torch=lambda z: z.square().sum(-1),
            render=lambda z: (z, z))
        with patch("image_benchmark.runner.directional_gradient_check", return_value={}) as check, redirect_stdout(io.StringIO()):
            result = preflight(problem)
            self.assertFalse(result["finite_difference_check"])
            check.assert_not_called()
            result = preflight(problem, finite_difference_check=True)
            self.assertTrue(result["finite_difference_check"])
            check.assert_called_once()
            problem.phi_and_grad = lambda z: (z.square().sum(-1), torch.full_like(z, float("nan")))
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                preflight(problem)

    def test_gradient_check_refines_for_curvature_and_rejects_wrong_derivative(self):
        from image_benchmark.runner import directional_gradient_check
        # A steep smooth transition fails the old fixed 1e-3 / 3e-4 checks.
        z = torch.zeros(1, 1, dtype=torch.float64)
        def potential(x):
            return (torch.tanh(1000*x) + 2).sum().reshape(1)
        gradient = torch.full_like(z, 1000.)
        with redirect_stdout(io.StringIO()):
            result = directional_gradient_check(potential, z, gradient)
        self.assertLess(result["directional_derivative_accepted_steps"][0], 3e-4)
        self.assertLess(result["directional_derivative_relative_error"], .01)
        # A linear potential has no truncation error: a wrong gradient must fail.
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "no two neighboring"):
            directional_gradient_check(lambda x: x.sum().reshape(1), z, torch.full_like(z, 1.2))

    def test_end_to_end_reports_observations_and_resume(self):
        from image_benchmark.runner import run
        from guided_diffusion.gaussian_diffusion import create_sampler
        def fake_imf(config,device):
            return TinyTransport(),TinyDecoder()
        def fake_dps(config,device):
            return TinyDenoiser(),create_sampler("ddpm",1000,"linear","epsilon","learned_range",False,True,False,"2")
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            checkpoint=directory/"test_weights_placeholder"
            checkpoint.write_bytes(b"unit-test fixture, not pretrained weights")
            image=directory/"fixture.png"
            rng=np.random.default_rng(10)
            Image.fromarray(rng.integers(0,256,(256,256,3),dtype=np.uint8)).save(image)
            config=copy.deepcopy(DEFAULT)
            config.update(device="cpu",outdir=str(directory/"out"),sampler_seeds=[0],examples=2)
            config["imf"]["checkpoint"]=str(checkpoint)
            config["dps"]["checkpoint"]=str(checkpoint)
            config["dps"]["steps"]=2
            config["dps"]["class_cond"]=True
            config["dps"]["batch_size"]=2
            config["spt"].update(replicas=3,chains=2,adapt_sweeps=2,burnin_sweeps=2,retained_per_chain=8,log_every=50)
            config["observations"]["phase_retrieval"]["oversample"]=1.
            manifest=[{"image_id":"fixture","image_path":str(image),"class_id":207}]
            with patch("image_benchmark.runner.load_imf",fake_imf),patch("image_benchmark.runner.load_dps",fake_dps),redirect_stdout(io.StringIO()):
                run(config,manifest)
                out=directory/"out"
                results=list(out.glob("runs/*/*/*/seed_*/result.json"))
                self.assertEqual(len(results),6)
                self.assertTrue((out/"summary.csv").is_file())
                self.assertTrue((out/"REPORT.md").is_file())
                old_times={str(p):p.stat().st_mtime_ns for p in results}
                observations={str(p):p.read_bytes() for p in out.glob("observations/*/*/observation.npz")}
                for path in results:
                    value=json.loads(path.read_text())["metrics"]
                    self.assertEqual(value["nominal_samples"],16)
                    self.assertGreater(value["runtime_sec"],0)
                    with np.load(path.parent/"traces.npz") as traces:
                        if value["method"]=="diffusion_dps":
                            self.assertIsNone(value["log_likelihood_ess"])
                            self.assertEqual(traces["log_likelihood"].shape,(16,))
                        else:
                            self.assertEqual(traces["latent"].shape,(8,2,4096))
                            self.assertEqual(traces["log_likelihood"].shape,(8,2))
                            self.assertAlmostEqual(value["log_likelihood_ess_per_sec"],value["log_likelihood_ess"]/value["runtime_sec"])
                    self.assertTrue((path.parent/"posterior_panel.png").is_file())
                run(config,manifest,resume=True)
                self.assertEqual(old_times,{str(p):p.stat().st_mtime_ns for p in results})
                self.assertEqual(observations,{str(p):p.read_bytes() for p in out.glob("observations/*/*/observation.npz")})
                bad=copy.deepcopy(config)
                bad["observations"]["super_resolution"]["sigma"]*=2
                with self.assertRaises(ValueError): run(bad,manifest,resume=True)


if __name__=="__main__":
    unittest.main()
