import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from image_benchmark.schedule import SweepSchedule, PHASES
from image_benchmark.samplers import ImageSPT
from image_benchmark.guidance_config import DEFAULT, validate
from run_final_guidance import sit_config, imf_config

class FinalSchedule(unittest.TestCase):
    def test_fixed_counts_and_zero_length_phases(self):
        s=SweepSchedule((0,2,3));phases=[]
        while (p:=s.next_phase(0)) is not None:
            phases.append(p);s.record(p,1)
        self.assertEqual(phases,['burnin']*2+['retained_sampling']*3)

    def test_time_phase_boundaries_reserve_whole_sweeps(self):
        s=SweepSchedule((250,250,1000),60)
        self.assertEqual(s.next_phase(0),'adaptation');s.record('adaptation',2)
        self.assertEqual(s.next_phase(8),'burnin');s.record('burnin',2)
        self.assertEqual(s.next_phase(18),'retained_sampling');s.record('retained_sampling',2)
        self.assertIsNone(s.next_phase(58))

    def test_budget_composition(self):
        c=copy.deepcopy(DEFAULT);c['spt']['chains']=8;c['prompts']=c['prompts'][:1];c['sampler_seeds']=[0]
        imf=imf_config(c);sit=sit_config(c,7200)
        self.assertEqual(imf['methods'],['imf_spt_pcn','imf_spt_hybrid'])
        self.assertIsNone(imf['time_budget'])
        self.assertEqual(sit['time_budget']['sampling_sec']+sit['time_budget']['overhead_sec'],10800)
        self.assertAlmostEqual(sum(sit['time_budget']['phase_fractions']),1)
        bad=copy.deepcopy(sit);bad['include_prior']=True
        with self.assertRaises(ValueError):validate(bad)

    def test_timed_sampler_freezes_adaptation_and_records_every_sweep(self):
        c=copy.deepcopy(DEFAULT['spt']);c.update(replicas=2,chains=2,adapt_sweeps=250,burnin_sweeps=250,retained_per_chain=1000)
        p=SimpleNamespace(dim=2,device=torch.device('cpu'),dtype=torch.float32,scenario='fixture',phi_torch=lambda z:z.square().sum(-1))
        sampler=ImageSPT(p,c,'sit_spt_pcn',12);rows=[]
        ticks=iter(np.arange(0,1000,.01))
        with patch('image_benchmark.samplers.time.perf_counter',side_effect=lambda:float(next(ticks))), patch.object(sampler,'_adapt',wraps=sampler._adapt) as adapt:
            result=sampler.run(diagnostics=True,sweep_callback=rows.append,
                               time_budget={'sampling_sec':4,'phase_fractions':[1/6,1/6,2/3],'remaining_total_sec':3604})
        self.assertEqual(len(rows),sum(result['phase_sweeps'].values()))
        self.assertEqual(adapt.call_count,result['phase_sweeps']['adaptation'])
        self.assertGreaterEqual(result['source'].shape[0],8)
        self.assertEqual(result['source'].shape[0],result['phase_sweeps']['retained_sampling'])
        self.assertTrue(all(r['sweep_sec']>0 for r in rows))
        phase_ids=[PHASES.index(r['phase']) for r in rows]
        self.assertEqual(phase_ids,sorted(phase_ids))
        self.assertEqual(len(result['telemetry']['sweep_sec']),len(rows))
        np.testing.assert_allclose(result['phi'],result['source'].square().sum(-1).numpy())

    def test_timed_guidance_exports_actual_draw_count(self):
        import tempfile,json,csv
        from pathlib import Path
        from test_guidance import FixtureTransport,FixtureDecoder,FixtureReward
        from image_benchmark.guidance import run
        original=ImageSPT.run
        def simulated_clock_run(sampler,*args,**kwargs):
            ticks=iter(np.arange(0,1000,.01))
            with patch('image_benchmark.samplers.time.perf_counter',side_effect=lambda:float(next(ticks))):
                return original(sampler,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ckpt=root/'fixture';ckpt.write_bytes(b'fixture')
            c=copy.deepcopy(DEFAULT)
            c.update(device='cpu',outdir=str(root/'out'),methods=['sit_spt_pcn'],include_prior=False,
                     sampler_seeds=[0],prior_samples=8,save_all_samples=True,save_chain_grids=True,
                     time_budget={'sampling_sec':4.,'overhead_sec':3600.,'phase_fractions':[1/6,1/6,2/3]})
            c['sit']['checkpoint']=str(ckpt);c['spt'].update(chains=2,replicas=2,retained_per_chain=1000)
            t=FixtureTransport();t.network_calls=t.network_state_evaluations=0
            with patch('image_benchmark.guidance.load_sit',return_value=(t,FixtureDecoder())), \
                 patch('image_benchmark.guidance.load_reward',return_value=(FixtureReward(),{'fixture':True})), \
                 patch.object(ImageSPT,'run',simulated_clock_run):
                run(c)
            result_path=next((root/'out').glob('runs/*/sit_spt_pcn/seed_0/result.json'))
            result=json.loads(result_path.read_text());d=result['details'];m=result['metrics']
            self.assertLess(m['retained_per_ladder'],1000)
            self.assertGreaterEqual(m['retained_per_ladder'],8)
            self.assertEqual(m['nominal_samples'],2*d['phase_sweeps']['retained_sampling'])
            directory=result_path.parent
            self.assertEqual(len(list((directory/'all_samples').glob('*.png'))),m['nominal_samples'])
            with (directory/'sweep_timings.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),sum(d['phase_sweeps'].values()))
            self.assertTrue((directory/'samples_all_ladders.png').exists())
            self.assertTrue((directory/'sampling_checkpoint.npz').exists())
