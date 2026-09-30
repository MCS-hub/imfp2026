#!/usr/bin/env python3
"""Class-conditional iMF/SiT + SPT-pCN or iMF + SPT-hybrid with CLIP reward."""
import argparse
import json
from pathlib import Path
from image_benchmark.guidance_config import read_config, validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/reward_guidance.json'))
    p.add_argument('--sit-checkpoint'); p.add_argument('--outdir'); p.add_argument('--device'); p.add_argument('--imf-checkpoint')
    p.add_argument('--class-id',type=int); p.add_argument('--class-name')
    p.add_argument('--prompt',help='Replace configured prompts with one prompt (ID: custom)')
    p.add_argument('--reward-strength',type=float); p.add_argument('--reward-model'); p.add_argument('--reward-revision')
    p.add_argument('--sampler-seeds',nargs='+',type=int); p.add_argument('--methods',nargs='+',choices=['imf_spt_pcn','imf_spt_hybrid','sit_spt_pcn'])
    p.add_argument('--finite-difference-check',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--pilot',action='store_true'); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--preflight-only',action='store_true'); p.add_argument('--resume',action='store_true')
    args = p.parse_args(); config = read_config(args.config)
    for key in ['outdir','device','class_id','class_name','sampler_seeds','methods','finite_difference_check']:
        if getattr(args,key) is not None: config[key] = getattr(args,key)
    if args.sit_checkpoint: config['sit']['checkpoint'] = args.sit_checkpoint
    if args.imf_checkpoint: config['imf']['checkpoint'] = args.imf_checkpoint
    for key in ['strength','model','revision']:
        if getattr(args,'reward_'+key) is not None: config['reward'][key] = getattr(args,'reward_'+key)
    if args.prompt is not None: config['prompts'] = [{'id':'custom','text':args.prompt}]
    if args.pilot:
        config['spt'].update(replicas=4,chains=2,adapt_sweeps=10,burnin_sweeps=10,retained_per_chain=16,betas=None,log_every=5)
        config.update(sampler_seeds=[0],prior_samples=16)
        print('PILOT: execution/tuning only; short traces cannot establish convergence.')
    validate(config)
    config['outdir'] = str(Path(config['outdir']).resolve())
    backend = 'sit' if config['methods'] == ['sit_spt_pcn'] else 'imf'
    config[backend]['checkpoint'] = str(Path(config[backend]['checkpoint']).expanduser().resolve())
    if not Path(config[backend]['checkpoint']).is_file(): p.error(f'Missing {backend} checkpoint')
    if args.dry_run:
        print(json.dumps(config,indent=2)); return
    from image_benchmark.guidance import run
    run(config,resume=args.resume,preflight_only=args.preflight_only)


if __name__ == '__main__': main()
