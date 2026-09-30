"""A separate protocol: no observation, image manifest, DPS, or reconstruction loss."""
import copy
import json
import math
import re
from .config import DEFAULT as IMAGE_DEFAULT, merge, validate as validate_image

DEFAULT = {
    'device':'cuda', 'outdir':'outs/golden_retriever_guidance_v1', 'class_id':207,
    'class_name':'golden retriever', 'methods':['imf_spt_pcn','imf_spt_hybrid'],
    'sampler_seeds':[0,1,2,3], 'likelihood_batch_size':1, 'examples':8,
    'include_prior':True, 'include_best_of_k':False, 'best_of_k':12, 'time_budget':None,
    'prior_samples':128, 'finite_difference_check':False, 'save_all_samples':False, 'save_chain_grids':False,
    'prompts':[{'id':'meadow','text':'A beautiful professional photograph of a golden retriever in a sunlit meadow, sharp focus, natural colors.'}],
    'imf':copy.deepcopy(IMAGE_DEFAULT['imf']),
    'sit':{'checkpoint':'checkpoints/SiT-XL-2-256x256.safetensors',
           'steps':125, 'solver':'heun', 'cfg_scale':4.0,
           'vae':'stabilityai/sd-vae-ft-mse', 'local_files_only':False},
    'spt':copy.deepcopy(IMAGE_DEFAULT['spt']),
    'reward':{'model':'openai/clip-vit-base-patch32', 'revision':'main',
              'local_files_only':False, 'strength':30.0},
}
DEFAULT['imf'].update(checkpoint='checkpoints/iMF-XL-2.pth',architecture='imfDiT_XL_2',activation_checkpointing=True)
DEFAULT['spt'].update(replicas=8,chains=4,adapt_sweeps=250,burnin_sweeps=250,
                      retained_per_chain=256,temperature_power=1.,initial_pcn_scale=.1,initial_hmc_epsilon=.01)


def read_config(path):
    return merge(copy.deepcopy(DEFAULT), json.loads(path.read_text()))


def validate(c):
    if type(c.get('include_prior', True)) is not bool:
        raise ValueError('include_prior must be boolean')
    if type(c.get('include_best_of_k', False)) is not bool:
        raise ValueError('include_best_of_k must be boolean')
    if type(c.get('best_of_k', 12)) is not int or c.get('best_of_k', 12) < 2:
        raise ValueError('best_of_k must be an integer >= 2')
    budget = c.get('time_budget')
    if budget is not None:
        if c['methods'] != ['sit_spt_pcn'] or len(c['sampler_seeds']) != 1 or len(c['prompts']) != 1 or c.get('include_prior',True) or c.get('include_best_of_k',False):
            raise ValueError('Timed SiT requires one prompt, one seed, and no prior run')
        if set(budget) != {'sampling_sec','overhead_sec','phase_fractions'}:
            raise ValueError('Invalid time-budget fields')
        if any(not math.isfinite(budget[k]) or budget[k] <= 0 for k in ('sampling_sec','overhead_sec')):
            raise ValueError('Time budgets must be positive and finite')
        fractions = budget['phase_fractions']
        if len(fractions) != 3 or any(not math.isfinite(f) or f <= 0 for f in fractions) or not math.isclose(sum(fractions),1.):
            raise ValueError('Three positive phase fractions must sum to one')

    if type(c.get('save_chain_grids', False)) is not bool:
        raise ValueError('save_chain_grids must be boolean')
    if c.get('save_chain_grids', False) and not c.get('save_all_samples', False):
        raise ValueError('save_chain_grids requires save_all_samples')
    if type(c.get('save_all_samples', False)) is not bool:
        raise ValueError('save_all_samples must be boolean')
    if type(c['class_id']) is not int or not 0 <= c['class_id'] < 1000:
        raise ValueError('class_id must be a zero-based ImageNet class ID')
    if not isinstance(c['class_name'],str) or not c['class_name'].strip():
        raise ValueError('class_name must be nonempty metadata; class_id controls conditioning')
    if not c['methods'] or any(m not in DEFAULT['methods'] + ['sit_spt_pcn'] for m in c['methods']):
        raise ValueError('Unsupported guidance method')
    if 'sit_spt_pcn' in c['methods'] and c['methods'] != ['sit_spt_pcn']:
        raise ValueError('Run SiT separately so its prior and protocol are unambiguous')
    sit = c['sit']
    if type(sit['steps']) is not int or sit['steps'] < 1 or sit['solver'] != 'heun':
        raise ValueError('SiT requires positive integer steps and solver=heun')
    if not math.isfinite(sit['cfg_scale']) or sit['cfg_scale'] < 1:
        raise ValueError('SiT cfg_scale must be finite and >= 1')
    if type(sit['local_files_only']) is not bool:
        raise ValueError('SiT local_files_only must be boolean')
    if 'sit_spt_pcn' in c['methods'] and c['finite_difference_check']:
        raise ValueError('SiT pCN does not use source gradients')
    if c.get('include_best_of_k', False) and c['methods'] == ['sit_spt_pcn']:
        raise ValueError('Best-of-K is an iMF prior baseline and cannot be combined with a SiT-only run')
    base = copy.deepcopy(IMAGE_DEFAULT)
    for key in ['imf','spt','methods','sampler_seeds','likelihood_batch_size','examples','finite_difference_check']:
        base[key] = c[key]
    base['methods'] = ['imf_spt_pcn' if m == 'sit_spt_pcn' else m for m in base['methods']]
    validate_image(base)
    for key in ['likelihood_batch_size','examples','prior_samples']:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['prior_samples'] < c['spt']['chains']:
        raise ValueError('prior_samples must be >= chains for equal-size terminal comparisons')
    if c['spt']['retained_per_chain'] < 8:
        raise ValueError('Use at least 8 retained draws per chain (more for meaningful diagnostics)')
    if type(c['imf']['steps']) is not int or type(c['imf']['activation_checkpointing']) is not bool:
        raise ValueError('Invalid iMF steps or checkpointing setting')
    if not math.isfinite(c['reward']['strength']) or c['reward']['strength'] <= 0:
        raise ValueError('reward.strength must be positive and finite; unguided samples are separate')
    if not isinstance(c['reward']['model'],str) or not c['reward']['model']:
        raise ValueError('Provide a reward model ID or local path')
    if type(c['reward']['local_files_only']) is not bool:
        raise ValueError('reward.local_files_only must be boolean')
    ids = set()
    if not isinstance(c['prompts'],list) or not c['prompts']:
        raise ValueError('Provide at least one prompt')
    for p in c['prompts']:
        if set(p) != {'id','text'} or not re.fullmatch('[A-Za-z0-9_-]+',p['id']) or p['id'] in ids or not isinstance(p['text'],str) or not p['text'].strip():
            raise ValueError('Prompts need unique safe IDs and nonempty text')
        ids.add(p['id'])
