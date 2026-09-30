"""Small invariant tests for the refactored SPT benchmark."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from curved_problem import CurvedObservationProblem, LearnedTransportPosterior
from spt_score import SPT, SamplerConfig, TorchSPT

if torch is not None:
    from learned_models import LearnedTransport, NetworkConfig, build_model


class ZeroPotentialProblem:
    scenario = "zero"

    @staticmethod
    def phi(z: np.ndarray) -> np.ndarray:
        return np.zeros(z.shape[:-1])

    @staticmethod
    def grad_phi(z: np.ndarray) -> np.ndarray:
        return np.zeros_like(z)


class BenchmarkInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SamplerConfig(
            dim=4,
            replicas=5,
            particles=4,
            adapt_sweeps=2,
            burnin_sweeps=2,
            retained_sweeps=3,
            hmc_steps=3,
        )
        self.problem = CurvedObservationProblem(
            "banana",
            self.cfg,
            sigma_y=0.4,
            observation_values=np.array([0.25, -0.5]),
            observation_tilt=0.35,
        )

    def test_tilted_operator_breaks_banana_branch_symmetry(self) -> None:
        uv = np.array([[1.0, 0.0], [-1.0, 0.0]])
        data = self.problem.transport_blocks(uv)
        prediction = self.problem.observe_blocks(data)
        self.assertAlmostEqual(prediction[0] - prediction[1], 0.70)

    def test_every_coordinate_is_active_and_every_pair_transformed(self):
        for dim in (2, 4, 32, 256):
            for scenario in ("banana", "sine"):
                with self.subTest(dim=dim, scenario=scenario):
                    cfg = SamplerConfig(dim=dim)
                    problem = CurvedObservationProblem(scenario, cfg, .4, np.zeros(dim // 2))
                    z = np.random.default_rng(6).normal(size=(2, 3, dim))
                    np.testing.assert_array_equal(problem.active(z), z)
                    self.assertEqual(problem.active_dim, dim)
                    self.assertEqual(problem.num_blocks, dim // 2)
                    x = problem.transport(z)
                    np.testing.assert_array_equal(x[..., ::2], z[..., ::2])
                    u, v = z[..., ::2], z[..., 1::2]
                    bend = .62 * (u**2 - 1) if scenario == "banana" else 1.05*np.sin(1.65*u)
                    tau = 0.30 if scenario == "banana" else 0.15
                    np.testing.assert_allclose(x[..., 1::2], tau*v + bend)
                    np.testing.assert_array_equal(problem.data_active(x), x)
                    self.assertFalse(hasattr(problem, "basis"))

    def test_invalid_dimensions_and_observations_rejected(self):
        for dim in (0, 1, 3, -2):
            with self.assertRaises(ValueError):
                CurvedObservationProblem("banana", SamplerConfig(dim=dim), .4, np.zeros(2))
        for observations in (np.zeros(1), np.zeros(3), np.array([0., np.nan])):
            with self.assertRaises(ValueError):
                CurvedObservationProblem("banana", self.cfg, .4, observations)

    def test_full_gradient_both_scenarios_high_dimension(self):
        rng = np.random.default_rng(31)
        for scenario in ("banana", "sine"):
            p = CurvedObservationProblem(scenario, SamplerConfig(dim=64), .3,
                                         rng.normal(size=32), observation_tilt=.35)
            z = rng.normal(size=(2, 3, 64))
            direction = rng.normal(size=z.shape)
            h = 1e-6
            finite = (p.phi(z+h*direction)-p.phi(z-h*direction))/(2*h)
            np.testing.assert_allclose(np.sum(p.grad_phi(z)*direction, axis=-1), finite,
                                       rtol=2e-6, atol=2e-5)
            np.testing.assert_allclose(p.phi(z), p.data_phi(p.transport(z)))
            # The final pair must influence the likelihood, not pass through untouched.
            changed = z.copy(); changed[..., -1] += 1.
            self.assertGreater(np.max(np.abs(p.phi(changed)-p.phi(z))), 1e-6)

    def test_numpy_sampler_returns_full_states(self):
        result = SPT(self.problem, self.cfg, "pcn", seed=17).run()
        self.assertEqual(result.active_trace.shape, (3, 4, 4))
        self.assertGreater(np.max(np.ptp(result.active_trace, axis=0)), 0.)

    def test_prior_and_checkpoint_identity_do_not_depend_on_observed_count(self):
        cfg = SamplerConfig(dim=64)
        z = np.random.default_rng(715).normal(size=(2, 3, 64))
        for scenario in ("banana", "sine"):
            problems = [CurvedObservationProblem(scenario, cfg, .4, np.zeros(m),
                                                observed_blocks=m) for m in (2, 8, 32)]
            expected = problems[0].transport(z)
            for p in problems:
                np.testing.assert_array_equal(p.transport(z), expected)
                self.assertEqual(p.prior_signature, problems[0].prior_signature)
                self.assertEqual(p.prior_dim, 64)
                self.assertEqual(p.num_blocks, 32)
                self.assertEqual(p.active(z).shape, (2, 3, 2*p.observed_blocks))
            # Even the final unobserved pair is curved, not copied from source.
            self.assertGreater(np.max(np.abs(expected[..., -1]-z[..., -1])), 1e-3)

    def test_partial_likelihood_and_full_gradient(self):
        rng = np.random.default_rng(712)
        for scenario in ("banana", "sine"):
            p = CurvedObservationProblem(scenario, SamplerConfig(dim=64), .4,
                                         np.array([2., -1.]), observation_tilt=.35,
                                         observed_blocks=2)
            z = rng.normal(size=(2, 3, 64))
            changed = z.copy(); changed[..., 4:] += rng.normal(size=changed[..., 4:].shape)
            np.testing.assert_array_equal(p.phi(changed), p.phi(z))
            x = p.transport(z)
            changed_x = x.copy(); changed_x[..., 4:] += 10.
            np.testing.assert_array_equal(p.data_phi(changed_x), p.data_phi(x))
            np.testing.assert_allclose(p.phi(z), p.data_phi(x))
            gradient = p.grad_phi(z)
            self.assertEqual(gradient.shape, z.shape)
            np.testing.assert_array_equal(gradient[..., 4:], 0.)
            direction = rng.normal(size=z.shape); h=1e-6
            finite = (p.phi(z+h*direction)-p.phi(z-h*direction))/(2*h)
            np.testing.assert_allclose(np.sum(gradient*direction, axis=-1), finite,
                                       rtol=2e-6, atol=2e-5)
            changed[..., 3] += 1.
            self.assertGreater(np.max(np.abs(p.phi(changed)-p.phi(z))), 1e-6)

    def test_observed_block_count_validation(self):
        for count in (0, -1, 3, 1.5):
            with self.assertRaises(ValueError):
                CurvedObservationProblem("banana", self.cfg, .4, np.zeros(2),
                                         observed_blocks=count)
        with self.assertRaises(ValueError):
            CurvedObservationProblem("banana", self.cfg, .4, np.zeros(2), observed_blocks=1)

    def test_partial_numpy_sampler_uses_full_state_and_returns_observed_prefix(self):
        cfg = SamplerConfig(dim=16, replicas=4, particles=3,
                            adapt_sweeps=2, burnin_sweeps=2, retained_sweeps=8)
        p = CurvedObservationProblem("banana", cfg, .4, np.array([2., -1.]),
                                     observed_blocks=2)
        result = SPT(p, cfg, "pcn", seed=44).run()
        self.assertEqual(result.active_trace.shape, (8, 3, 4))
        self.assertGreater(np.max(np.ptp(result.active_trace, axis=0)), 0.)

    def test_likelihood_gradient_matches_directional_difference(self) -> None:
        rng = np.random.default_rng(91)
        z = rng.normal(size=(3, self.cfg.dim))
        direction = rng.normal(size=z.shape)
        epsilon = 1e-6
        finite_difference = (
            self.problem.phi(z + epsilon * direction)
            - self.problem.phi(z - epsilon * direction)
        ) / (2.0 * epsilon)
        analytic = np.sum(self.problem.grad_phi(z) * direction, axis=-1)
        np.testing.assert_allclose(analytic, finite_difference, rtol=2e-6, atol=2e-6)

    def test_pcn_accepts_every_proposal_at_zero_potential(self) -> None:
        sampler = SPT(ZeroPotentialProblem(), self.cfg, "pcn", seed=7)
        z = np.zeros((self.cfg.particles, self.cfg.dim))
        phi = np.zeros(self.cfg.particles)
        proposal, proposal_phi, accepted = sampler._pcn(
            z, phi, beta=0.73, eta=0.4
        )
        self.assertTrue(np.all(accepted))
        self.assertEqual(proposal.shape, z.shape)
        np.testing.assert_array_equal(proposal_phi, 0.0)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_numpy_and_torch_exact_transport_agree(self) -> None:
        rng = np.random.default_rng(413)
        for scenario in ("banana", "sine"):
            for dim in (2, 64):
                problem = CurvedObservationProblem(scenario, SamplerConfig(dim=dim), .4,
                                                   np.zeros(min(2, dim // 2)), observation_tilt=.35,
                                                   observed_blocks=min(2, dim // 2))
                source = rng.normal(size=(2, 3, dim))
                numpy_data = problem.transport(source)
                torch_source = torch.tensor(source, dtype=torch.float64, requires_grad=True)
                torch_data = problem.transport_torch(torch_source)
                np.testing.assert_allclose(torch_data.detach().numpy(), numpy_data, atol=1e-12)
                phi = problem.data_phi_torch(torch_data)
                np.testing.assert_allclose(phi.detach().numpy(), problem.phi(source), atol=1e-10)
                grad = torch.autograd.grad(phi.sum(), torch_source)[0]
                np.testing.assert_allclose(grad.detach().numpy(), problem.grad_phi(source), atol=1e-10)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_learned_posterior_retains_cross_coordinate_gradients(self):
        class CoupledTransport:
            device = torch.device("cpu")
            dtype = torch.float64
            @staticmethod
            def __call__(z):
                x = z.clone()
                x[..., 1] = z[..., 1] + z[..., -1]
                return x
        p = CurvedObservationProblem("banana", SamplerConfig(dim=8), .4,
                                     np.array([1.]), observed_blocks=1)
        posterior = LearnedTransportPosterior(p, CoupledTransport())
        z = torch.zeros(2, 8, dtype=torch.float64)
        _, grad = posterior.phi_and_grad(z)
        self.assertEqual(grad.shape, z.shape)
        self.assertTrue(torch.all(grad[..., -1] != 0))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_learned_networks_share_backbone_shape(self) -> None:
        config = NetworkConfig(dim=self.cfg.dim, hidden=32, depth=2)
        state = torch.randn(5, self.cfg.dim)
        time = torch.rand(5)
        imf = build_model("imf", config)
        fm = build_model("fm", config)
        diffusion = build_model("diffusion", config)
        average = imf(state, torch.zeros_like(time), time)
        velocity = imf.get_v(state, time)
        self.assertEqual(average.shape, state.shape)
        self.assertEqual(velocity.shape, state.shape)
        self.assertEqual(fm(state, time).shape, state.shape)
        self.assertEqual(diffusion(state, time).shape, state.shape)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_learned_transport_preserves_all_batch_axes(self) -> None:
        config = NetworkConfig(dim=self.cfg.dim, hidden=32, depth=2)
        source = torch.randn(5, 4, self.cfg.dim)
        imf = LearnedTransport(
            "imf", build_model("imf", config), steps=1, solver="euler"
        )
        fm = LearnedTransport(
            "fm", build_model("fm", config), steps=2, solver="rk4"
        )
        self.assertEqual(imf(source).shape, source.shape)
        self.assertEqual(fm(source).shape, source.shape)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_torch_spt_returns_fixed_count_per_chain(self) -> None:
        class IdentityTransport:
            device = torch.device("cpu")
            dtype = torch.float64

            @staticmethod
            def __call__(z: torch.Tensor) -> torch.Tensor:
                return z

        config = SamplerConfig(
            dim=4,
            replicas=4,
            particles=3,
            adapt_sweeps=2,
            burnin_sweeps=2,
            retained_sweeps=5,
            hmc_steps=2,
            beta_hmc=1.0,
            cold_hmc_probability=1.0,
        )
        problem = CurvedObservationProblem(
            "banana", config,
            sigma_y=0.4, observation_values=np.array([0.25, -0.5]),
        )
        posterior = LearnedTransportPosterior(problem, IdentityTransport())
        result = TorchSPT(posterior, config, "pcn", seed=17).run()
        self.assertEqual(result.active_trace.shape, (5, 3, 4))
        self.assertEqual(result.data_active_trace.shape, (5, 3, 4))
        self.assertEqual(result.retained_per_chain, 5)
        self.assertEqual(result.retained_chains, 3)


    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_torch_spt_cpu_retains_independent_snapshots(self):
        class IdentityTransport:
            device = torch.device("cpu")
            dtype = torch.float64

            @staticmethod
            def __call__(z):
                return z

        class NumberedSweeps(TorchSPT):
            # All replicas receive the same sweep number, so swaps cannot
            # change the expected snapshots. Exercise real in-place storage.
            def _pcn(self, z, phi, beta, eta):
                calls = getattr(self, "test_calls", 0)
                self.test_calls = calls + 1
                return (torch.full_like(z, 1 + calls // self.cfg.replicas),
                        torch.zeros_like(phi), torch.ones_like(phi, dtype=torch.bool))

            def _split_hmc(self, z, phi, beta, epsilon):
                return self._pcn(z, phi, beta, epsilon)

        cfg = SamplerConfig(dim=4, replicas=4, particles=3,
                            adapt_sweeps=0, burnin_sweeps=0,
                            retained_sweeps=5, thin=2,
                            beta_hmc=1.0, cold_hmc_probability=1.0)
        problem = CurvedObservationProblem("banana", cfg, .4, np.array([.25, -.5]))
        posterior = LearnedTransportPosterior(problem, IdentityTransport())
        expected = np.broadcast_to(np.arange(1, 10, 2)[:, None, None], (5, 3, 4))
        for method in ("pcn", "full_split_hmc_mixture"):
            with self.subTest(method=method):
                result = NumberedSweeps(posterior, cfg, method, seed=17).run()
                np.testing.assert_array_equal(result.active_trace, expected)
                np.testing.assert_array_equal(result.data_active_trace, expected)


if __name__ == "__main__":
    unittest.main()
