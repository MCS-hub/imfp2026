"""Guidance tests use explicitly artificial weights/data, never ImageNet results."""
import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from image_benchmark.rewards import CLIPReward, RewardPotential
from image_benchmark.guidance_config import DEFAULT, validate
from image_benchmark.guidance import best_of_k_sources, feature_diversity, round_trips, run
from image_benchmark.samplers import ImageSPT


torch.set_num_threads(1)


class FixtureTransport(nn.Module):
    def forward(self,z,class_id):
        assert class_id == 207
        return (.5*z).reshape(-1,4,32,32)


class FixtureDecoder(nn.Module):
    def forward(self,z):
        return F.interpolate(z[:,:3].tanh(),size=(256,256),mode='nearest')


class FixtureReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('text_feature',torch.tensor([[1.,0.,0.,0.]]))
    def set_prompt(self,text):
        self.prompt=text
    def features(self,images):
        x=images.mean((-1,-2))
        return F.normalize(torch.cat([x,torch.ones_like(x[:,:1])],-1),dim=-1)
    def forward(self,images):
        return (self.features(images)*self.text_feature).sum(-1)


class Guidance(unittest.TestCase):
    def test_actual_clip_architecture_input_gradient_and_batch_invariance(self):
        from transformers import CLIPConfig, CLIPModel
        torch.manual_seed(32)
        config=CLIPConfig(text_config={'vocab_size':16,'hidden_size':16,'intermediate_size':32,
            'num_hidden_layers':1,'num_attention_heads':2,'max_position_embeddings':8,
            'bos_token_id':1,'eos_token_id':2,'pad_token_id':0},
            vision_config={'image_size':8,'patch_size':4,'hidden_size':16,'intermediate_size':32,
            'num_hidden_layers':1,'num_attention_heads':2},projection_dim=8)
        model=CLIPModel(config).eval()
        def tokenize(*a,**kw):
            return {'input_ids':torch.tensor([[1,4,2]]),'attention_mask':torch.ones(1,3,dtype=torch.long)}
        reward=CLIPReward(model,tokenize,image_size=8)
        reward.set_prompt('artificial fixture')
        x=(torch.rand(2,3,16,16)-.5).requires_grad_(True)
        scores=reward(x)
        gradient=torch.autograd.grad(scores.sum(),x)[0]
        self.assertGreater(float(gradient.norm()),0)
        torch.testing.assert_close(scores,torch.cat([reward(y[None]) for y in x]),atol=1e-6,rtol=1e-5)
        direction=gradient/gradient.norm()
        with torch.no_grad():
            fd=(reward(x+1e-3*direction).sum()-reward(x-1e-3*direction).sum())/2e-3
        torch.testing.assert_close(fd,(gradient*direction).sum(),rtol=.01,atol=1e-4)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in reward.parameters()))
        self.assertTrue(torch.isfinite(reward(torch.full_like(x,2.))).all())

    def test_positive_reward_sign_and_diversity(self):
        potential=RewardPotential(lambda x:x.flatten(1).mean(1),3.)
        x=torch.tensor([[1.],[2.]],requires_grad=True)
        torch.testing.assert_close(potential(x),torch.tensor([-3.,-6.],dtype=torch.float64))
        torch.testing.assert_close(torch.autograd.grad(potential(x).sum(),x)[0],torch.full_like(x,-3.))
        rng=np.random.default_rng(4); f=rng.normal(size=(7,4)); f/=np.linalg.norm(f,axis=1,keepdims=True)
        expected=np.mean([np.sum((f[i]-f[j])**2) for i in range(7) for j in range(i+1,7)])
        self.assertAlmostEqual(feature_diversity(f),expected)
        self.assertAlmostEqual(feature_diversity(np.ones((4,3))),0)
        labels=np.array([[[0],[1]],[[1],[0]],[[0],[1]]])
        self.assertEqual(round_trips(labels).sum(),1)

    def test_guidance_validation(self):
        c=copy.deepcopy(DEFAULT); validate(c)
        for key,value in [('class_id',1000),('methods',['diffusion_dps']),('prompts',[{'id':'../bad','text':'test'}])]:
            bad=copy.deepcopy(c); bad[key]=value
            with self.assertRaises(ValueError): validate(bad)
        bad=copy.deepcopy(c); bad['reward']['strength']=-1
        with self.assertRaises(ValueError): validate(bad)

    def test_best_of_k_selects_independent_group_maxima(self):
        draws,chains,k,dim,seed=3,2,12,4,19
        generator=torch.Generator(device='cpu').manual_seed(seed+200000)
        candidates=torch.randn(draws*chains,k,dim,generator=generator)
        expected=candidates[torch.arange(draws*chains),candidates[...,0].argmax(1)]
        problem=SimpleNamespace(dim=dim,device=torch.device('cpu'),dtype=torch.float32,
                                phi_torch=lambda z:-z[...,0].double())
        actual=best_of_k_sources(problem,draws,chains,k,seed)
        torch.testing.assert_close(actual.reshape(-1,dim),expected)

    def test_best_of_k_run_reports_all_candidates_and_selected_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'fixture_weights'; checkpoint.write_bytes(b'artificial test fixture')
            c=copy.deepcopy(DEFAULT)
            c.update(device='cpu',outdir=str(root/'out'),methods=['imf_spt_pcn'],include_prior=False,
                     include_best_of_k=True,sampler_seeds=[0])
            c['imf']['checkpoint']=str(checkpoint)
            c['spt'].update(replicas=2,chains=2,adapt_sweeps=1,burnin_sweeps=1,retained_per_chain=8)
            with patch('image_benchmark.guidance.load_imf',return_value=(FixtureTransport(),FixtureDecoder())), patch(
                    'image_benchmark.guidance.load_reward',return_value=(FixtureReward(),{'fixture':True})):
                with redirect_stdout(io.StringIO()):
                    run(c)
            result_path=next((root/'out').glob('runs/*/imf_best_of_12/seed_0/result.json'))
            metrics=json.loads(result_path.read_text())['metrics']
            self.assertEqual(metrics['best_of_k'],12)
            self.assertEqual(metrics['candidate_groups'],16)
            self.assertEqual(metrics['candidate_samples'],192)
            self.assertEqual(metrics['nominal_samples'],16)
            self.assertEqual(metrics['terminal_sample_count'],16)
            self.assertNotIn('reward_ess',metrics)

    def test_positive_linear_tilt_gaussian_recovery_and_telemetry(self):
        problem=SimpleNamespace(dim=2,device=torch.device('cpu'),dtype=torch.float64,scenario='fixture',
            phi_torch=lambda z:-.4*z[...,0],
            phi_and_grad=lambda z:(-.4*z[...,0],torch.stack([torch.full_like(z[...,0],-.4),torch.zeros_like(z[...,0])],-1)))
        c=copy.deepcopy(DEFAULT['spt']); c.update(replicas=3,chains=4,adapt_sweeps=20,burnin_sweeps=100,
            retained_per_chain=800,initial_hmc_epsilon=.3,initial_pcn_scale=.5,hmc_steps=4)
        for method in DEFAULT['methods']:
            result=ImageSPT(problem,c,method,7).run(diagnostics=True)
            x=result['source'].numpy()
            np.testing.assert_allclose(x.mean((0,1)),[.4,0],atol=.12)
            np.testing.assert_allclose(x.var((0,1)),[1,1],atol=.18)
            np.testing.assert_allclose(result['phi'],-.4*x[:,:,0])
            self.assertEqual(result['telemetry']['potential'].shape,(920,3,4))
            self.assertEqual(result['telemetry']['local_acceptance_by_chain'].shape,(3,4))

    def test_end_to_end_paired_target_outputs_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'fixture_weights'; checkpoint.write_bytes(b'not pretrained: artificial test fixture')
            c=copy.deepcopy(DEFAULT); c.update(device='cpu',outdir=str(root/'out'),sampler_seeds=[0],prior_samples=8,examples=3,save_all_samples=True,save_chain_grids=True)
            c['imf']['checkpoint']=str(checkpoint)
            c['spt'].update(replicas=3,chains=2,adapt_sweeps=2,burnin_sweeps=2,retained_per_chain=8)
            with patch('image_benchmark.guidance.load_imf',return_value=(FixtureTransport(),FixtureDecoder())), \
                 patch('image_benchmark.guidance.load_reward',return_value=(FixtureReward(),{'artificial_test_fixture':True})), redirect_stdout(io.StringIO()):
                run(c)
                out=root/'out'; results=list(out.glob('runs/*/*/seed_*/result.json'))
                self.assertEqual(len(results),3)
                self.assertTrue((out/'paired.csv').exists())
                initials=[]
                for p in results:
                    m=json.loads(p.read_text())['metrics']
                    self.assertEqual(m['terminal_sample_count'],2)
                    with np.load(p.parent/'traces.npz') as t:
                        self.assertEqual(t['source'].shape[-1],4096)
                        if m['method']!='prior':
                            self.assertEqual(t['reward'].shape,(8,2))
                    if m['method']!='prior':
                        with np.load(p.parent/'telemetry.npz') as t:
                            initials.append(t['initial_cold_source'])
                            self.assertEqual(t['potential'].shape,(12,3,2))
                    self.assertTrue((p.parent/'samples.png').exists())
                    exported = sorted((p.parent/'all_samples').glob('*.png'))
                    self.assertEqual(len(exported), m['nominal_samples'])
                    with np.load(p.parent/'traces.npz') as t:
                        n_draws, n_chains = t['reward'].shape
                    self.assertEqual(len(list(p.parent.glob('chain_*_all_samples.png'))), n_chains)
                    for chain in range(n_chains):
                        with Image.open(p.parent/f'chain_{chain:03d}_all_samples.png') as grid:
                            self.assertEqual(grid.size, (256*min(8,n_draws),308*((n_draws+7)//8)))
                            for draw in range(n_draws):
                                x,y=(draw%8)*256,(draw//8)*308
                                with Image.open(p.parent/'all_samples'/f'draw_{draw:06d}_chain_{chain:03d}.png') as single:
                                    np.testing.assert_array_equal(np.asarray(grid.crop((x,y+52,x+256,y+308))),np.asarray(single))
                    with (p.parent/'all_samples'/'index.csv').open() as f:
                        rows = list(csv.DictReader(f))
                    with np.load(p.parent/'traces.npz') as t:
                        np.testing.assert_allclose([float(row['reward']) for row in rows], t['reward'].reshape(-1))
                        for i, row in enumerate(rows):
                            self.assertEqual((int(row['draw']), int(row['chain'])), divmod(i, t['reward'].shape[1]))
                            self.assertTrue((p.parent/'all_samples'/row['file']).is_file())
                np.testing.assert_array_equal(initials[0],initials[1])
                times={p:p.stat().st_mtime_ns for p in results}
                run(c,resume=True)
                self.assertEqual(times,{p:p.stat().st_mtime_ns for p in results})
                bad=copy.deepcopy(c); bad['reward']['strength']+=1
                with self.assertRaisesRegex(ValueError,'mismatch'): run(bad,resume=True)



class PanelSelection(unittest.TestCase):
    def test_balanced_ladder_rows(self):
        from image_benchmark.guidance import panel_indices
        small=panel_indices(1000,8,4,2).reshape(2,4)
        np.testing.assert_array_equal(small//8,[[0,333,666,999]]*2)
        np.testing.assert_array_equal(small%8,[[0]*4,[1]*4])
        full=panel_indices(1000,8,8).reshape(8,8)
        np.testing.assert_array_equal(full//8,np.tile(np.linspace(0,999,8,dtype=int),(8,1)))
        np.testing.assert_array_equal(full%8,np.repeat(np.arange(8)[:,None],8,axis=1))

if __name__=='__main__': unittest.main()
