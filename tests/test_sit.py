"""Analytic transport and kernel tests, independent of pretrained image quality."""
import copy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from image_benchmark.sit import SiTTransport, SiTDecoder
from image_benchmark.guidance_config import DEFAULT, validate
from image_benchmark.samplers import ImageSPT

class Velocity(nn.Module):
    def forward(self, x, t, y):
        return x + t[:,None,None,None]
    def forward_with_cfg(self, x, t, y, cfg_scale):
        # Same duplication and three-channel CFG semantics as upstream.
        half = x[:len(x)//2]
        raw = self.forward(torch.cat([half,half]),t,y) + y[:,None,None,None]/1000
        cond, uncond = raw[:,:3].chunk(2)
        guided = uncond + cfg_scale*(cond-uncond)
        return torch.cat([torch.cat([guided,guided]),raw[:,3:]],1)

class SiTTests(unittest.TestCase):
    def test_heun_direction_accuracy_determinism_and_nfe(self):
        z=torch.zeros(2,4096)
        tr=SiTTransport(Velocity(),steps=100,cfg_scale=1)
        result=tr(z,207)
        # x'=x+t, x(0)=0 => x(1)=e-2.
        torch.testing.assert_close(result,torch.full_like(result,torch.exp(torch.tensor(1.)).item()-2),atol=5e-5,rtol=0)
        self.assertEqual(tr.network_calls,200)
        self.assertEqual(tr.network_state_evaluations,400)
        torch.testing.assert_close(result,tr(z,207),atol=0,rtol=0)
        torch.testing.assert_close(result[:1],tr(z[:1],207),atol=0,rtol=0)

    def test_cfg_matches_full_doubled_state_reference(self):
        z=torch.randn(2,4096); model=Velocity(); steps=5; dt=1/steps
        x=torch.cat([z.reshape(-1,4,32,32)]*2)
        labels=torch.tensor([207,207,1000,1000])
        for i in range(steps):
            k1=model.forward_with_cfg(x,torch.full((4,),i*dt),labels,4.)
            k2=model.forward_with_cfg(x+dt*k1,torch.full((4,),(i+1)*dt),labels,4.)
            x=x+dt/2*(k1+k2)
        tr=SiTTransport(model,steps=steps,cfg_scale=4.)
        torch.testing.assert_close(tr(z,207),x[:2],atol=0,rtol=0)
        self.assertEqual(tr.network_state_evaluations,40)

    def test_decoder_and_pcn_without_gradients(self):
        class VAE(nn.Module):
            def decode(self,x): return SimpleNamespace(sample=x)
        x=torch.ones(1,4,32,32)
        torch.testing.assert_close(SiTDecoder(VAE())(x),x/.18215)
        c=copy.deepcopy(DEFAULT); c['methods']=['sit_spt_pcn']; validate(c)
        s=c['spt']; s.update(replicas=2,chains=2,adapt_sweeps=1,burnin_sweeps=1,retained_per_chain=8)
        p=SimpleNamespace(dim=2,device=torch.device('cpu'),dtype=torch.float32,scenario='fixture',phi_torch=lambda z:z.square().sum(-1))
        a=ImageSPT(p,s,'sit_spt_pcn',42).run(diagnostics=True)
        b=ImageSPT(p,s,'imf_spt_pcn',42).run(diagnostics=True)
        torch.testing.assert_close(a['source'],b['source'],atol=0,rtol=0)
        self.assertEqual(a['source'].shape,(8,2,2))

    def test_sit_guidance_end_to_end(self):
        import tempfile, json
        from pathlib import Path
        from unittest.mock import patch
        from test_guidance import FixtureTransport, FixtureDecoder, FixtureReward
        from image_benchmark.guidance import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ckpt=root/'fixture'; ckpt.write_bytes(b'artificial test fixture')
            c=copy.deepcopy(DEFAULT)
            c.update(device='cpu',methods=['sit_spt_pcn'],outdir=str(root/'out'),
                     sampler_seeds=[0],prior_samples=8,save_all_samples=True,save_chain_grids=True)
            c['sit']['checkpoint']=str(ckpt)
            c['spt'].update(replicas=2,chains=2,adapt_sweeps=1,burnin_sweeps=1,retained_per_chain=8)
            transport=FixtureTransport()
            transport.network_calls=transport.network_state_evaluations=0
            with patch('image_benchmark.guidance.load_sit',return_value=(transport,FixtureDecoder())), \
                 patch('image_benchmark.guidance.load_reward',return_value=(FixtureReward(),{'fixture':True})):
                run(c)
            paths=list((root/'out').glob('runs/*/*/seed_0/result.json'))
            self.assertEqual(len(paths),2)
            self.assertEqual({json.loads(p.read_text())['metrics']['method'] for p in paths},{'prior','sit_spt_pcn'})
            contract=json.loads((root/'out/protocol.json').read_text())['contract']
            self.assertIn('sit_checkpoint_sha256',contract)
            self.assertNotIn('imf_checkpoint_sha256',contract)
            self.assertEqual(len(list((root/'out').glob('runs/*/sit_spt_pcn/seed_0/chain_*_all_samples.png'))),2)
