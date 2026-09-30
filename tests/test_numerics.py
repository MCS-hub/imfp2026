"""CPU checks for invariance, gradients, sampler semantics, and ESS."""
import copy
import math
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from image_benchmark.config import DEFAULT, validate
from image_benchmark.models import ImageProblem, IMFTransport, LatentDecoder, LATENT_MEAN, LATENT_STD
from image_benchmark.observations import SuperResolution, FourierMagnitude, PixelLikelihood
from image_benchmark.samplers import ImageSPT, split_trajectory, gaussian_energy, swap_log_ratio
from image_benchmark.diagnostics import coordinate_ess, split_rhat
from image_benchmark.dps import BatchPosteriorSampling

torch.set_num_threads(1)


class ToyTransport(nn.Module):
    def forward(self, source, class_id):
        return source.reshape(-1,1,2,2)


class ToyDecoder(nn.Module):
    def forward(self, latent):
        return torch.nn.functional.interpolate(latent.repeat(1,3,1,1),size=(8,8),mode="bilinear",align_corners=False)


class GaussianProblem:
    device = torch.device("cpu")
    dtype = torch.float64
    dim = 2
    scenario = "analytic Gaussian"
    target = torch.tensor([0.6,-0.4],dtype=torch.float64)
    sigma = .8

    def phi_torch(self,z):
        return .5*((z-self.target)/self.sigma).square().sum(-1)

    def phi_and_grad(self,z):
        return self.phi_torch(z), (z-self.target)/self.sigma**2


class Numerics(unittest.TestCase):
    def test_linear_operator(self):
        x = torch.randn(2,3,16,16,dtype=torch.float64)
        y = torch.randn_like(x)
        a = SuperResolution(4)
        torch.testing.assert_close(a(2*x-3*y),2*a(x)-3*a(y))

    def test_nonlinear_operator(self):
        x = torch.randn(1,3,8,8,dtype=torch.float64,requires_grad=True)
        a = FourierMagnitude(2,1e-6)
        torch.testing.assert_close(a(x),a(-x))
        y = torch.randn_like(x)
        self.assertFalse(torch.allclose(a(x+y),a(x)+a(y)))
        self.assertTrue(torch.autograd.gradcheck(a,(x,),eps=1e-6,atol=1e-5,fast_mode=True))
        zero = torch.zeros_like(x,requires_grad=True)
        self.assertTrue(torch.isfinite(torch.autograd.grad(a(zero).sum(),zero)[0]).all())

    def test_pixel_likelihood_gradient_and_microbatching(self):
        torch.manual_seed(17)
        for operator in [SuperResolution(2),FourierMagnitude(1.5,1e-4)]:
            observation = operator(torch.randn(1,3,8,8,dtype=torch.float64))
            likelihood = PixelLikelihood(operator,observation,.2)
            p = ImageProblem(ToyTransport(),ToyDecoder(),likelihood,0,"cpu",1,4,torch.float64)
            z = torch.randn(2,3,4,dtype=torch.float64)
            phi,grad = p.phi_and_grad(z)
            z2 = z.clone().requires_grad_(True)
            direct = p.phi_torch(z2)
            g2 = torch.autograd.grad(direct.sum(),z2)[0]
            torch.testing.assert_close(phi,direct)
            torch.testing.assert_close(grad,g2)
            direction = torch.randn_like(z)
            eps = 1e-6
            numeric = (p.phi_torch(z+eps*direction)-p.phi_torch(z-eps*direction))/(2*eps)
            torch.testing.assert_close(numeric,(grad*direction).sum(-1),atol=1e-5,rtol=1e-5)
            p.batch_size=4
            phi4,g4 = p.phi_and_grad(z)
            torch.testing.assert_close(phi4,phi)
            torch.testing.assert_close(g4,grad)

    def test_decoder_normalization(self):
        class VAE(nn.Module):
            def decode(self,x):
                self.last=x
                return SimpleNamespace(sample=x[:,:3])
        vae=VAE()
        decoder=LatentDecoder(vae).double()
        z=torch.randn(2,4,32,32,dtype=torch.float64,requires_grad=True)
        image=decoder(z)
        expected=z*torch.tensor(LATENT_STD).view(1,4,1,1)+torch.tensor(LATENT_MEAN).view(1,4,1,1)
        torch.testing.assert_close(vae.last,expected)
        self.assertGreater(float(torch.autograd.grad(image.sum(),z)[0].norm()),0)

    def test_split_reversible_and_volume_preserving(self):
        problem=GaussianProblem()
        z=torch.tensor([[.2,-.6]],dtype=torch.float64)
        p=torch.tensor([[.3,.8]],dtype=torch.float64)
        z1,p1,_=split_trajectory(z,p,.7,.12,5,problem.phi_and_grad)
        z2,p2,_=split_trajectory(z1,-p1,.7,.12,5,problem.phi_and_grad)
        torch.testing.assert_close(z2,z,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(p2,-p,atol=1e-12,rtol=1e-12)
        def map_state(state):
            zz,pp,_=split_trajectory(state[:2][None],state[2:][None],.7,.12,5,problem.phi_and_grad)
            return torch.cat([zz.flatten(),pp.flatten()])
        jac=torch.autograd.functional.jacobian(map_state,torch.cat([z.flatten(),p.flatten()]))
        self.assertAlmostEqual(float(torch.linalg.det(jac)),1.,places=10)
        z0,p0,_=split_trajectory(z,p,0.,.2,6,problem.phi_and_grad)
        torch.testing.assert_close(gaussian_energy(z0,p0),gaussian_energy(z,p))

    def test_swap_sign(self):
        # Moving a lower-Phi state from hot to cold must be favorable.
        self.assertGreater(swap_log_ratio(.1,1.,torch.tensor(2.),torch.tensor(7.)),0)

    def test_stationary_gaussian_and_trace_cache(self):
        problem=GaussianProblem()
        s=copy.deepcopy(DEFAULT["spt"])
        s.update(replicas=4,chains=4,adapt_sweeps=100,burnin_sweeps=300,
                 retained_per_chain=1800,initial_pcn_scale=.5,initial_hmc_epsilon=.2,hmc_steps=4)
        expected_mean=problem.target.numpy()/(1+problem.sigma**2)
        expected_var=problem.sigma**2/(1+problem.sigma**2)
        for method in ["imf_spt_pcn","imf_spt_hybrid"]:
            result=ImageSPT(problem,s,method,seed=94).run()
            samples=result["source"].numpy()
            np.testing.assert_allclose(result["phi"],problem.phi_torch(result["source"]).numpy(),atol=1e-12)
            np.testing.assert_allclose(samples.mean((0,1)),expected_mean,atol=.065)
            np.testing.assert_allclose(samples.var((0,1)),expected_var,atol=.065)
            self.assertGreater(float(samples.std()),.1)
            self.assertEqual(samples.shape,(1800,4,2))

    def test_ess_iid_ar1_and_stuck(self):
        rng=np.random.default_rng(24)
        noise=rng.normal(size=(6000,4,3))
        iid=coordinate_ess(noise)
        self.assertTrue(np.all(iid>0.75*6000*4))
        correlated=np.zeros_like(noise)
        for t in range(1,len(noise)):
            correlated[t]=.9*correlated[t-1]+noise[t]
        estimate=coordinate_ess(correlated[500:])
        expected=5500*4*(1-.9)/(1+.9)
        np.testing.assert_allclose(estimate,expected,rtol=.35)
        np.testing.assert_array_equal(coordinate_ess(np.zeros((100,4,2))),0)
        shifted=np.zeros((100,4,2));shifted[:,1]=5
        self.assertTrue(np.isinf(split_rhat(shifted)).all())

    def test_dps_guidance_is_per_image(self):
        x=torch.randn(3,3,8,8,requires_grad=True)
        measurement=torch.zeros(1,3,4,4)
        condition=BatchPosteriorSampling(SuperResolution(2),None,scale=1.)
        batch_grad,_=condition.grad_and_value(x,.3*x,measurement)
        for i in range(3):
            single=x[i:i+1].detach().requires_grad_(True)
            single_grad,_=condition.grad_and_value(single,.3*single,measurement)
            torch.testing.assert_close(single_grad,batch_grad[i:i+1])

    def test_native_ddpm_batch_zero_step(self):
        from guided_diffusion.gaussian_diffusion import create_sampler
        sampler=create_sampler("ddpm",1000,"linear","epsilon","learned_range",False,True,False,"8")
        def model(x,t,y=None):
            self.assertTrue(torch.equal(y,torch.tensor([2,3,4])))
            return torch.cat([torch.zeros_like(x),torch.zeros_like(x)],1)
        x=torch.randn(3,3,8,8)
        t=torch.tensor([0,0,0])
        kwargs={"y":torch.tensor([2,3,4])}
        expected=sampler.p_mean_variance(model,x,t,model_kwargs=kwargs)["mean"]
        actual=sampler.p_sample(model,x,t,model_kwargs=kwargs)["sample"]
        torch.testing.assert_close(actual,expected)

    def test_config_rejects_degenerate_ladder(self):
        config=copy.deepcopy(DEFAULT)
        validate(config)
        config["spt"]["betas"]=[0]*23+[1]
        with self.assertRaises(ValueError): validate(config)


if __name__=="__main__":
    unittest.main()
