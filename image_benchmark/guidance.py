"""Class-conditional iMF Gibbs guidance: identical target, two local kernels."""
import json
import csv
import math
import platform
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .models import ImageProblem, load_imf
from .sit import load_sit, preflight_sit
from .samplers import ImageSPT, synchronize
from .rewards import RewardPotential, load_reward, state_hash
from .diagnostics import coordinate_ess, split_rhat
from .reporting import write_json, write_csv, aggregate, rgb_image
from .runner import file_hash, log, preflight
from .vendor import ROOT


def feature_diversity(features):
    """Mean squared distance over unordered pairs of normalized CLIP features.

    All-pairs diversity is descriptive: retained MCMC draws can be correlated.
    """
    x = np.asarray(features,dtype=np.float64)
    n = len(x)
    if n < 2:
        return None
    return float(max(0., 2*(n*np.square(x).sum()-np.square(x.sum(0)).sum())/(n*(n-1))))


def round_trips(labels):
    """Count hot->cold->hot trips by persistent replica label and ladder."""
    _, replicas, chains = labels.shape
    phase = np.zeros((replicas,chains),dtype=int)
    trips = np.zeros_like(phase)
    for frame in labels:
        for c in range(chains):
            hot,cold = frame[0,c],frame[-1,c]
            if phase[hot,c] == 2:
                trips[hot,c] += 1
            phase[hot,c] = 1
            if phase[cold,c] == 1:
                phase[cold,c] = 2
    return trips


LABEL_HEIGHT = 52
CELL_HEIGHT = 256 + LABEL_HEIGHT


def panel_font():
    for name in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(name, 18)
        except OSError:
            pass
    return ImageFont.load_default(size=18)


def panel_label(painter, xy, label, font):
    # Two lines keep four-digit draw numbers legible without shrinking text.
    painter.multiline_text(xy, label.replace(': R=', '\nR='), font=font,
                           fill=(20, 20, 20), spacing=3)


def panel_indices(draws, chains, per_chain, max_chains=None):
    selected = np.linspace(0, draws-1, min(per_chain, draws), dtype=int)
    count = chains if max_chains is None else min(chains, max_chains)
    return np.array([int(draw)*chains+chain for chain in range(count)
                     for draw in selected], dtype=int)


def best_of_k_sources(problem, draws, chains, k, seed, group_batch_size=32):
    groups = draws * chains
    generator = torch.Generator(device=problem.device).manual_seed(seed + 200000)
    winners = []
    with torch.no_grad():
        for start in range(0, groups, group_batch_size):
            count = min(group_batch_size, groups - start)
            candidates = torch.randn(count, k, problem.dim, generator=generator,
                                     device=problem.device, dtype=problem.dtype)
            potential = problem.phi_torch(candidates.reshape(-1, problem.dim)).reshape(count, k)
            best = potential.argmin(dim=1)
            group = torch.arange(count, device=problem.device)
            winners.append(candidates[group, best].detach().cpu())
    return torch.cat(winners).reshape(draws, chains, problem.dim)


def save_panel(path, images, labels, columns=4):
    columns = min(columns, len(images))
    panel = Image.new('RGB',(256*columns,CELL_HEIGHT*math.ceil(len(images)/columns)),'white')
    draw = ImageDraw.Draw(panel)
    font = panel_font()
    for i,(im,label) in enumerate(zip(images,labels)):
        x,y = (i%columns)*256,(i//columns)*CELL_HEIGHT
        panel_label(draw, (x+8,y+5), label, font)
        picture = im.convert('RGB') if isinstance(im, Image.Image) else rgb_image(im)
        panel.paste(picture.resize((256,256)),(x,y+LABEL_HEIGHT))
    panel.save(path)


@torch.no_grad()
def render_outputs(problem, source, reward, examples, export_directory=None):
    export_sec, export_rows = 0., []
    if export_directory is not None:
        export_directory = Path(export_directory)
        export_directory.mkdir(parents=True, exist_ok=True)
    flat = source.reshape(-1,problem.dim)
    draws, chains = source.shape[:2]
    small_indices = panel_indices(draws, chains, 4, max_chains=2)
    overview_indices = panel_indices(draws, chains, 8)
    chosen = set(small_indices) | set(overview_indices)
    latent, features, scores, pictures, indices, radii = [],[],[],[],[],[]
    best_score, best_image = -math.inf,None
    offset = 0
    for batch in flat.split(problem.batch_size):
        batch = batch.to(problem.device)
        z,image = problem.render(batch)
        feat = reward.features(image)
        score = (feat*reward.text_feature).sum(-1)
        if not torch.isfinite(image).all() or not torch.isfinite(score).all() or not torch.isfinite(feat).all():
            raise FloatingPointError('Nonfinite reward or generated image')
        latent.append(z.flatten(1).cpu().numpy()); features.append(feat.cpu().numpy())
        scores.append(score.cpu().numpy()); radii.append(batch.double().norm(dim=-1).cpu().numpy())
        for k in range(len(batch)):
            if offset+k in chosen:
                pictures.append(image[k].cpu().numpy()); indices.append(offset+k)
            if float(score[k]) > best_score:
                best_score = float(score[k]); best_image = image[k].cpu().numpy()
        if export_directory is not None:
            images_cpu = image.cpu().numpy()
            export_start = time.perf_counter()
            for k, im in enumerate(images_cpu):
                index = offset+k
                draw, chain = divmod(index, chains)
                filename = f'draw_{draw:06d}_chain_{chain:03d}.png'
                rgb_image(im).save(export_directory/filename)
                export_rows.append({'file':filename,'index':index,'draw':draw,'chain':chain,
                                    'reward':float(scores[-1][k])})
            export_sec += time.perf_counter()-export_start
        offset += len(batch)
    if export_directory is not None:
        export_start = time.perf_counter()
        write_csv(export_directory/'index.csv', export_rows)
        export_sec += time.perf_counter()-export_start
    selected_images = dict(zip(indices, pictures))
    return dict(export_sec=export_sec,latent=np.concatenate(latent),features=np.concatenate(features),reward=np.concatenate(scores),
                radius=np.concatenate(radii),examples=np.stack([selected_images[i] for i in small_indices]),example_indices=small_indices,
                overview=np.stack([selected_images[i] for i in overview_indices]),overview_indices=overview_indices,best=best_image)


@torch.no_grad()
def export_all_samples(problem, source, scores, directory):
    """Export every retained state, including repeats; run outside benchmark timing."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    chains = source.shape[1]
    rows = []
    offset = 0
    for batch in source.reshape(-1, problem.dim).split(problem.batch_size):
        _, images = problem.render(batch.to(problem.device))
        if not torch.isfinite(images).all():
            raise FloatingPointError('Nonfinite image during sample export')
        for k, im in enumerate(images.cpu().numpy()):
            index = offset + k
            draw, chain = divmod(index, chains)
            filename = f'draw_{draw:06d}_chain_{chain:03d}.png'
            rgb_image(im).save(directory / filename)
            rows.append({'file': filename, 'index': index, 'draw': draw,
                         'chain': chain, 'reward': float(scores[index])})
        offset += len(batch)
    write_csv(directory / 'index.csv', rows)


def save_chain_grids(directory, scores, draws, chains):
    """One chronological contact sheet per ladder, built from exported PNGs."""
    directory = Path(directory)
    columns = min(8, draws)
    for chain in range(chains):
        panel = Image.new('RGB', (256*columns, CELL_HEIGHT*math.ceil(draws/columns)), 'white')
        painter = ImageDraw.Draw(panel)
        font = panel_font()
        for draw in range(draws):
            x, y = (draw % columns)*256, (draw // columns)*CELL_HEIGHT
            score = float(scores[draw*chains+chain])
            panel_label(painter, (x+8,y+5), f'draw {draw}, chain {chain}: R={score:.3f}', font)
            with Image.open(directory / f'draw_{draw:06d}_chain_{chain:03d}.png') as im:
                panel.paste(im.convert('RGB').resize((256, 256)), (x, y+LABEL_HEIGHT))
        panel.save(directory.parent / f'chain_{chain:03d}_all_samples.png')


def summarize_outputs(arrays, terminal_indices):
    r = arrays['reward']; terminal = r[terminal_indices]
    return {'reward_mean':float(r.mean()),'reward_std':float(r.std(ddof=1)),
            'reward_p05':float(np.quantile(r,.05)), 'reward_p95':float(np.quantile(r,.95)),
            'reward_max':float(r.max()), 'nominal_samples':len(r),
            'source_radius_mean':float(arrays['radius'].mean()),
            'clip_diversity_all':feature_diversity(arrays['features']),
            'terminal_sample_count':len(terminal), 'terminal_reward_mean':float(terminal.mean()),
            'terminal_reward_best':float(terminal.max()),
            'clip_diversity_terminal':feature_diversity(arrays['features'][terminal_indices])}


def report(outdir):
    rows = [json.loads(p.read_text())['metrics'] for p in sorted(outdir.glob('runs/*/*/seed_*/result.json'))]
    write_csv(outdir/'by_seed.csv',rows)
    write_csv(outdir/'summary.csv',aggregate(rows,['prompt_id','method']))
    pairs = []
    lookup = {(r['prompt_id'],r['seed'],r['method']):r for r in rows}
    for (prompt,seed,method),p in lookup.items():
        if method != 'imf_spt_pcn': continue
        h = lookup.get((prompt,seed,'imf_spt_hybrid'))
        if h is None: continue
        row = {'prompt_id':prompt,'seed':seed}
        for metric in ['reward_mean','terminal_reward_mean','terminal_reward_best','clip_diversity_terminal','runtime_sec','reward_ess_per_sec']:
            if p.get(metric) is not None and h.get(metric) is not None:
                row[metric+'_hybrid_minus_pcn'] = h[metric]-p[metric]
        pairs.append(row)
    write_csv(outdir/'paired.csv',pairs)
    lines = ['# Class-conditional reward guidance','',
             'Frozen class-conditional transport (iMF or SiT, as recorded in protocol.json); text enters only the CLIP reward. This is not ImageReward or text-conditioned generation.',
             '', '| Prompt | Method | Seed | Mean cosine reward | Terminal best | Reward ESS/s | Hours |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        ess = r.get('reward_ess_per_sec')
        cell = f'{ess:.4g}' if ess is not None else 'N/A'
        lines.append(f"| {r['prompt_id']} | {r['method']} | {r['seed']} | {r['reward_mean']:.4f} | {r['terminal_reward_best']:.4f} | {cell} | {r['runtime_sec']/3600:.3f} |")
    lines += ['', 'Terminal scores use one final cold state per independent ladder; the prior uses IID draws, and Best-of-K selects the highest-reward state independently within each group of K prior draws. '
              'samples.png shows the first two ladders, four equally spaced draws per row; samples_all_ladders.png shows every ladder, eight equally spaced draws per row. Repeated states are preserved. best.png is explicitly selected by reward.',
              'CLIP diversity uses normalized embeddings and measures feature variation, not calibrated uncertainty or independent sample count.',
              'ESS uses the existing initial-positive-sequence estimator; split R-hat is ordinary, not rank-normalized. '
              'Runtime includes initialization, adaptation, burn-in, all replicas, and final rendering/scoring; excludes loading, preflight, hashing, diagnostics, file writing and image export (sample_export_sec); all retained images are exported from the same final rendering pass.',
              'summary.csv contains descriptive mean and sample SD across seeds; paired.csv contains matched-seed differences. '
              'A higher CLIP score does not establish aesthetic quality or class fidelity. Inspect images and convergence diagnostics.']
    (outdir/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(config, resume=False, preflight_only=False):
    run_start = time.perf_counter()
    device = torch.device(config['device']); outdir = Path(config['outdir']).resolve()
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on a GPU host')
    if outdir.exists() and not resume:
        raise FileExistsError('Choose a new output directory or use --resume for the identical protocol')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    begin = time.perf_counter()
    backend = 'sit' if config['methods'] == ['sit_spt_pcn'] else 'imf'
    transport, decoder = (load_sit if backend == 'sit' else load_imf)(config[backend],device)
    reward, reward_metadata = load_reward(config['reward'],device)
    synchronize(device); load_sec = time.perf_counter()-begin
    files = [ROOT/'run_reward_guidance.py'] + sorted((ROOT/'image_benchmark').glob('*.py'))
    files += sorted((ROOT/'vendor/imeanflow-torch').rglob('*.py'))
    files += [ROOT/'vendor/imfp_scripts/spt_score.py']
    if backend == 'sit': files += sorted((ROOT/'vendor/sit').glob('*.py'))
    contract = {'config':config,backend+'_checkpoint_sha256':file_hash(config[backend]['checkpoint']),
                'decoder_weights_sha256':state_hash(decoder),'reward':reward_metadata,
                'code_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in files},
                'target':'exp(-||z||^2/2 + beta*tau*cosine(CLIP_image(D(S(z;c))), CLIP_text(prompt))); beta in [0,1]',
                'initialization':'IID N(0,I) at every replica; paired methods share initial states through identical seeds; no warm start',
                'trace_axes':'draw, independent ladder, coordinate; telemetry: sweep, temperature, ladder',
                'environment':{'torch':torch.__version__,'python':platform.python_version(),
                               'transformers':version('transformers'),'diffusers':version('diffusers')}}
    protocol_path = outdir/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['contract'] != contract:
            raise ValueError('Resume protocol/model/code mismatch; choose a new output directory')
    elif resume and outdir.exists():
        raise ValueError('Existing directory has no guidance protocol')
    outdir.mkdir(parents=True,exist_ok=True)
    if not protocol_path.exists():
        write_json(protocol_path,{'contract':contract,'model_load_sec':load_sec})
    best_of_method = f"imf_best_of_{config['best_of_k']}"
    log(f"{backend} steps={config[backend]['steps']}; class={config['class_id']}; reward strength={config['reward']['strength']}")
    for prompt in config['prompts']:
        reward.set_prompt(prompt['text'])
        potential = RewardPotential(reward,config['reward']['strength'])
        problem = ImageProblem(transport,decoder,potential,config['class_id'],device,config['likelihood_batch_size'])
        log(f"Preflight reward: {prompt['id']}")
        info = preflight_sit(problem) if backend == 'sit' else preflight(problem,config['finite_difference_check'])
        write_json(outdir/f"preflight_{prompt['id']}.json",info)
        np.save(outdir/f"text_feature_{prompt['id']}.npy",reward.text_feature.detach().cpu().numpy())
        if preflight_only: continue
        for seed in config['sampler_seeds']:
            methods = (['prior'] if config.get('include_prior',True) else [])
            if config.get('include_best_of_k', False):
                methods.append(best_of_method)
            for method in methods+config['methods']:
                path = outdir/'runs'/prompt['id']/method/f'seed_{seed}'
                if (path/'result.json').exists(): continue
                path.mkdir(parents=True,exist_ok=True)
                log(f"Running {prompt['id']} / {method} / seed {seed}")
                if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
                details, telemetry = {},{}
                calls_before = getattr(transport, 'network_calls', 0)
                states_before = getattr(transport, 'network_state_evaluations', 0)
                if method == 'prior':
                    synchronize(device); start = time.perf_counter()
                    rng = torch.Generator(device=device).manual_seed(seed+100000)
                    source = torch.randn(config['prior_samples'],1,problem.dim,generator=rng,device=device).cpu()
                    synchronize(device); sampling_sec = time.perf_counter()-start
                elif method == best_of_method:
                    draws = config['spt']['retained_per_chain']
                    chains = config['spt']['chains']
                    candidate_count = draws * chains * config['best_of_k']
                    synchronize(device); start = time.perf_counter()
                    source = best_of_k_sources(problem, draws, chains, config['best_of_k'], seed)
                    synchronize(device); sampling_sec = time.perf_counter()-start
                    details = {'best_of_k':config['best_of_k'],
                               'candidate_groups':draws*chains,
                               'candidate_samples':candidate_count}
                else:
                    budget = config.get('time_budget')
                    if budget is not None:
                        budget = dict(budget)
                        budget['remaining_total_sec'] = budget['sampling_sec']+budget['overhead_sec']-(time.perf_counter()-run_start)
                    with (path/'sweep_timings.csv').open('w', newline='') as timing_file:
                        writer = csv.DictWriter(timing_file,fieldnames=['sweep','phase','phase_sweep','sweep_sec','elapsed_sampling_sec','cold_phi_mean','retained_draws'])
                        writer.writeheader(); timing_file.flush()
                        def record_sweep(row):
                            writer.writerow(row); timing_file.flush()
                        result = ImageSPT(problem,config['spt'],method,seed).run(
                            progress=log,diagnostics=True,sweep_callback=record_sweep,time_budget=budget)
                    source = result.pop('source'); telemetry = result.pop('telemetry')
                    sampling_sec = result['sampling_sec']; details = result
                    np.savez_compressed(path/'sampling_checkpoint.npz',source=source.numpy(),phi=details['phi'])
                    np.savez_compressed(path/'telemetry.npz',**telemetry)
                    write_json(path/'sampling_details.json',{k:v for k,v in details.items() if k != 'phi'})
                synchronize(device); start = time.perf_counter()
                arrays = render_outputs(problem,source,reward,config['examples'],
                                        export_directory=path/'all_samples' if config.get('save_all_samples',False) else None)
                synchronize(device); render_sec = time.perf_counter()-start-arrays['export_sec']
                runtime = sampling_sec+render_sec
                draws,chains,dim = source.shape
                if method == 'prior':
                    terminal = np.arange(config['spt']['chains'])
                elif method == best_of_method:
                    terminal = np.arange(len(arrays['reward']))
                else:
                    terminal = np.arange(len(arrays['reward'])-chains,len(arrays['reward']))
                metrics = {'prompt_id':prompt['id'],'method':method,'seed':seed,
                           **summarize_outputs(arrays,terminal), 'runtime_sec':runtime,'sampling_sec':sampling_sec,
                           'final_render_sec':render_sec, 'samples_per_sec':len(arrays['reward'])/runtime,
                           'final_reward_evaluations':len(arrays['reward'])}
                if method == best_of_method:
                    metrics.update(best_of_k=config['best_of_k'],
                                   candidate_groups=details['candidate_groups'],
                                   candidate_samples=details['candidate_samples'])
                if backend == 'sit':
                    metrics.update(transport_nfe_per_mapping=2*config['sit']['steps'],
                                   transport_network_calls=transport.network_calls-calls_before,
                                   transport_network_state_evaluations=transport.network_state_evaluations-states_before)
                metrics['peak_gpu_allocated_mb'] = torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None
                traces = {'source':source.numpy(),'latent':arrays['latent'].reshape(draws,chains,-1),
                          'reward':arrays['reward'].reshape(draws,chains),'clip_features':arrays['features'].reshape(draws,chains,-1),
                          'source_radius':arrays['radius'].reshape(draws,chains)}
                if method not in ('prior', best_of_method):
                    np.testing.assert_allclose(-config['reward']['strength']*traces['reward'],details.pop('phi'),rtol=1e-5,atol=1e-5)
                    metrics.update(sweep_sec_mean=details['sweep_sec_mean'],sweep_sec_median=details['sweep_sec_median'],
                                   completed_sweeps=sum(details['phase_sweeps'].values()),retained_per_ladder=draws)
                    ess = coordinate_ess(traces['reward'])[0]; latent_ess = coordinate_ess(traces['latent'])
                    metrics.update(reward_ess=float(ess),reward_ess_per_sec=float(ess/runtime),
                                   reward_split_rhat=float(split_rhat(traces['reward'])[0]),
                                   latent_ess_mean=float(latent_ess.mean()),latent_ess_mean_per_sec=float(latent_ess.mean()/runtime),
                                   latent_split_rhat_max=float(split_rhat(traces['latent']).max()),
                                   cold_acceptance=details['kernel']['local_acceptance'][-1],
                                   minimum_swap_acceptance=min(details['kernel']['swap_acceptance']))
                    counts = round_trips(telemetry['replica_labels'][details['phase_sweeps']['adaptation']:])
                    details['round_trips_by_replica_and_chain'] = counts.tolist()
                    details['local_acceptance_by_chain'] = telemetry['local_acceptance_by_chain'].tolist()
                    details['swap_acceptance_by_chain'] = telemetry['swap_acceptance_by_chain'].tolist()
                    metrics['round_trips_total'] = int(counts.sum())
                    repeats = np.all(np.diff(source.numpy(),axis=0)==0,axis=-1).mean(0)
                    details['retained_repeat_fraction_by_chain'] = repeats.tolist()
                    np.savez_compressed(path/'telemetry.npz',**telemetry)
                np.savez_compressed(path/'traces.npz',**traces)
                np.savez_compressed(path/'examples.npz',images=arrays['examples'],indices=arrays['example_indices'],best=arrays['best'],
                                    overview=arrays['overview'],overview_indices=arrays['overview_indices'])
                labels = [f"draw {i//chains}, chain {i%chains}: R={arrays['reward'][i]:.3f}" for i in arrays['example_indices']]
                save_panel(path/'samples.png',arrays['examples'],labels,columns=min(4,draws))
                overview_labels = [f"draw {i//chains}, chain {i%chains}: R={arrays['reward'][i]:.3f}" for i in arrays['overview_indices']]
                save_panel(path/'samples_all_ladders.png',arrays['overview'],overview_labels,columns=min(8,draws))
                rgb_image(arrays['best']).save(path/'best.png')
                if config.get('save_all_samples', False):
                    synchronize(device); export_start = time.perf_counter()
                    if config.get('save_chain_grids', False):
                        save_chain_grids(path/'all_samples', arrays['reward'], draws, chains)
                    synchronize(device)
                    metrics['sample_export_sec'] = arrays['export_sec']+time.perf_counter()-export_start
                if config.get('time_budget') is not None:
                    metrics['total_run_wall_sec'] = time.perf_counter()-run_start
                    metrics['total_budget_sec'] = config['time_budget']['sampling_sec']+config['time_budget']['overhead_sec']
                    metrics['budget_overrun_sec'] = max(0.,metrics['total_run_wall_sec']-metrics['total_budget_sec'])
                write_json(path/'result.json',{'metrics':metrics,'details':details})
                report(outdir)
                log(f"Saved {method}: mean reward={metrics['reward_mean']:.4f}, time={runtime:.1f}s")
