"""Shell objective and optimization output checks using artificial CPU fixtures."""
import copy
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import run_image_optimization as opt
from image_benchmark.config import DEFAULT


class GaussianFit:
    dim = 3
    device = torch.device('cpu')
    likelihood = SimpleNamespace(measurement_dim=3, sigma=1.)

    def phi_torch(self, z):
        return .5*z.square().sum(-1)


class FixtureTransport(nn.Module):
    def forward(self, z, class_id):
        return (.5*z).reshape(-1,4,32,32)


class FixtureDecoder(nn.Module):
    def forward(self, z):
        return torch.nn.functional.interpolate(z[:,:3].tanh(), size=(256,256), mode='nearest')


class Optimization(unittest.TestCase):
    def test_shell_mode_and_gradient(self):
        z = torch.tensor([[math.sqrt(3),0.,0.,0.]], dtype=torch.float64, requires_grad=True)
        self.assertAlmostEqual(float(opt.gaussian_shell_penalty(z).detach()),0.,places=12)
        self.assertLess(float(torch.autograd.grad(opt.gaussian_shell_penalty(z).sum(),z)[0].norm()),1e-12)
        for radius, sign in [(1.,-1),(3.,1)]:
            x = torch.tensor([[radius,0.,0.,0.]],dtype=torch.float64,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(opt.gaussian_shell_penalty,(x,)))
            self.assertGreater(sign*float(torch.autograd.grad(opt.gaussian_shell_penalty(x).sum(),x)[0][0,0]),0)

    def test_shell_prevents_radial_collapse_and_selects_best(self):
        initial = torch.tensor([[3.,0.,0.]], dtype=torch.float64)
        rows = []
        regularized = opt.optimize_source(GaussianFit(), initial, 400, .03, 10., rows.append)
        unregularized = opt.optimize_source(GaussianFit(), initial, 400, .03, 0.)
        self.assertLess(unregularized['best']['source_radius'],.02)
        self.assertAlmostEqual(regularized['best']['source_radius'],math.sqrt(2),delta=.04)
        self.assertLess(regularized['best']['phi'],rows[0]['phi'])
        self.assertEqual(regularized['best']['objective'],min(r['objective'] for r in rows))
        self.assertEqual(len(rows),401)
        self.assertEqual(rows[-1]['iteration'],400)
        self.assertTrue(torch.equal(initial,torch.tensor([[3.,0.,0.]],dtype=torch.float64)))

    def test_cli_saved_observation_and_outputs(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source'; source.mkdir()
            weights = root/'fixture.pt'; weights.write_bytes(b'artificial fixture')
            config = copy.deepcopy(DEFAULT)
            config['device'] = 'cpu'; config['tasks'] = ['super_resolution']
            config['imf']['checkpoint'] = str(weights)
            contract = {'config': config, 'manifest': [{'image_id':'fixture','class_id':207}],
                        'provenance': {'checkpoints': {'imf':opt.file_hash(weights)}, 'code':{}}}
            (source/'protocol.json').write_text(json.dumps({'contract':contract}))
            observation_dir = source/'observations/super_resolution/fixture'; observation_dir.mkdir(parents=True)
            np.savez(observation_dir/'observation.npz',truth=np.zeros((1,3,256,256),np.float32),
                     measurement=np.zeros((1,3,64,64),np.float32))
            out = root/'out'
            argv = ['run_image_optimization.py','--run-dir',str(source),'--outdir',str(out),
                    '--starts','2','--iterations','2']
            with patch('sys.argv',argv), patch.object(opt,'load_imf',return_value=(FixtureTransport(),FixtureDecoder())), redirect_stdout(io.StringIO()):
                opt.main()
            self.assertTrue((out/'preflight.json').is_file())
            self.assertFalse(json.loads((out/'preflight.json').read_text())['finite_difference_check'])
            self.assertTrue((out/'optimization_panel.png').is_file())
            best = json.loads((out/'best.json').read_text())
            results = [json.loads(p.read_text())['best'] for p in out.glob('start_*/result.json')]
            self.assertEqual(best['objective'],min(r['objective'] for r in results))
            for p in out.glob('start_*/reconstruction.npz'):
                with np.load(p) as saved:
                    z = torch.from_numpy(saved['source'])
                    expected = FixtureDecoder()(FixtureTransport()(z,207)).numpy()
                    np.testing.assert_allclose(saved['image'],expected)
                    result = json.loads((p.parent/'result.json').read_text())['best']
                    self.assertAlmostEqual(result['source_radius'],float(z.double().norm()))
                    self.assertEqual(saved['source'].shape,(1,4096))
            with patch('sys.argv',argv), self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                opt.main()


if __name__ == '__main__':
    unittest.main()
