#!/usr/bin/env python3
"""Sequential final comparison: full iMF pCN, full iMF hybrid, then budgeted SiT."""
import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from image_benchmark.guidance_config import read_config, validate


def sit_config(base, hybrid_sampling_sec):
    c=copy.deepcopy(base)
    c.update(methods=['sit_spt_pcn'],include_prior=False,
             outdir=str(Path(base['outdir'])/'sit_final'))
    c['time_budget']={'sampling_sec':float(hybrid_sampling_sec),'overhead_sec':3600.,
                      'phase_fractions':[1/6,1/6,2/3]}
    # SiT batches two ladders (four model states with CFG) as in the verified pilot.
    c['likelihood_batch_size']=2
    validate(c)
    return c


def imf_config(base):
    c=copy.deepcopy(base)
    c.update(methods=['imf_spt_pcn','imf_spt_hybrid'],include_prior=False,
             include_best_of_k=True,
             time_budget=None,outdir=str(Path(base['outdir'])/'imf_final'))
    validate(c)
    return c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); base=read_config(args.config)
    base['outdir']=str(Path(base['outdir']).resolve())
    validate(base)
    if len(base['prompts'])!=1 or len(base['sampler_seeds'])!=1:
        p.error('Each final job must contain exactly one prompt and one sampler seed')
    if base['imf']['steps']!=1 or base['spt']['chains']!=8 or base['spt']['replicas']!=8:
        p.error('Final protocol requires 1-step iMF, 8 ladders, and 8 replicas')
    for backend in ('imf','sit'):
        base[backend]['checkpoint']=str(Path(base[backend]['checkpoint']).resolve())
        if not Path(base[backend]['checkpoint']).is_file(): p.error(f'Missing {backend} checkpoint')
    imf=imf_config(base)
    if args.dry_run:
        print(json.dumps({'imf':imf,'sit_budget_rule':'hybrid sampling_sec + 3600s overhead',
                          'sit_phase_fractions':[1/6,1/6,2/3],
                          'sit_template':sit_config(base,3600.)},indent=2));return
    root=Path(base['outdir'])
    if root.exists() and not args.resume: raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    def launch(config,filename):
        path=root/filename
        if path.exists() and json.loads(path.read_text())!=config:
            raise ValueError(f'Existing protocol differs: {path}')
        path.write_text(json.dumps(config,indent=2)+'\n')
        command=[sys.executable,'-u','run_reward_guidance.py','--config',str(path)]
        if args.resume and (Path(config['outdir'])/'protocol.json').exists():command.append('--resume')
        print('Launching '+ ' '.join(command),flush=True)
        subprocess.run(command,check=True)
    launch(imf,'imf_final.resolved.json')
    prompt=base['prompts'][0]['id'];seed=base['sampler_seeds'][0]
    hybrid_path=Path(imf['outdir'])/'runs'/prompt/'imf_spt_hybrid'/f'seed_{seed}'/'result.json'
    hybrid=json.loads(hybrid_path.read_text())
    seconds=hybrid['metrics']['sampling_sec']
    sit=sit_config(base,seconds)
    budget={'hybrid_result':str(hybrid_path),'hybrid_sampling_sec':seconds,
            'hybrid_runtime_with_render_sec':hybrid['metrics']['runtime_sec'],
            'sit_total_budget_sec':seconds+3600,'sit_sampling_budget_sec':seconds,
            'extra_overhead_sec':3600,'phase_fractions':[1/6,1/6,2/3],
            'planned_phase_seconds_before_initialization':[seconds/6,seconds/6,seconds*2/3],
            'rounding':'Only complete sweeps; reserves estimated rendering cost if needed'}
    (root/'sit_budget_final.json').write_text(json.dumps(budget,indent=2)+'\n')
    print('Measured hybrid timing / SiT budget: '+json.dumps(budget),flush=True)
    launch(sit,'sit_final.resolved.json')
    rows=[]
    best_of_method=f"imf_best_of_{base['best_of_k']}"
    for backend,methods in [('imf_final',['imf_spt_pcn','imf_spt_hybrid',best_of_method]),('sit_final',['sit_spt_pcn'])]:
        for method in methods:
            result=json.loads((root/backend/'runs'/prompt/method/f'seed_{seed}'/'result.json').read_text())
            m=result['metrics'];d=result['details']
            phases=d.get('phase_sweeps',{})
            rows.append({'class_id':base['class_id'],'class_name':base['class_name'],'prompt':base['prompts'][0]['text'],
                         'method':method,'sampling_sec':m['sampling_sec'],'runtime_sec':m['runtime_sec'],
                         'export_sec':m.get('sample_export_sec',0),
                         'mean_sweep_sec':d.get('sweep_sec_mean'),'median_sweep_sec':d.get('sweep_sec_median'),
                         'adapt_sweeps':phases.get('adaptation'),'burnin_sweeps':phases.get('burnin'),
                         'retained_sweeps':phases.get('retained_sampling'),
                         'retained_samples':m['nominal_samples'],'best_of_k':m.get('best_of_k'),
                         'candidate_samples':m.get('candidate_samples'),'reward_mean':m['reward_mean'],
                         'reward_best':m['reward_max'],'reward_rhat':m.get('reward_split_rhat'),
                         'cold_acceptance':m.get('cold_acceptance')})
    with (root/'comparison_final.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (root/'completion_final.json').write_text(json.dumps({'completed':True,'job_wall_sec':time.perf_counter()-start},indent=2)+'\n')
    print('FINAL COMPARISON COMPLETE: '+str(root),flush=True)

if __name__=='__main__':main()
