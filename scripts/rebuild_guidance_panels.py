#!/usr/bin/env python3
"""Rebuild ladder-balanced panels from completed runs' exported PNGs; no model calls."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from image_benchmark.guidance import panel_indices, save_panel


def rebuild(directory):
    directory = Path(directory)
    if not (directory/'result.json').exists():
        return False
    index = directory/'all_samples/index.csv'
    rows = list(csv.DictReader(index.open()))
    draws = 1+max(int(r['draw']) for r in rows)
    chains = 1+max(int(r['chain']) for r in rows)
    lookup = {int(r['draw'])*chains+int(r['chain']):r for r in rows}
    if set(lookup) != set(range(draws*chains)):
        raise ValueError(f'Incomplete sample export: {directory}')
    selections = {}
    arrays = {}
    for filename, per_chain, limit, prefix in [('samples.png',4,2,'small'),
                                              ('samples_all_ladders.png',8,None,'overview')]:
        indices = panel_indices(draws,chains,per_chain,limit)
        images, labels = [], []
        for i in indices:
            row = lookup[int(i)]
            with Image.open(directory/'all_samples'/row['file']) as im:
                images.append(im.convert('RGB').copy())
            labels.append(f"draw {row['draw']}, chain {row['chain']}: R={float(row['reward']):.3f}")
        target = directory/filename
        backup = target.with_name(target.stem+'.original.png')
        if target.exists() and not backup.exists(): shutil.copy2(target,backup)
        save_panel(target,images,labels,columns=min(per_chain,draws))
        selections[filename] = [lookup[int(i)] for i in indices]
        arrays[prefix] = np.stack([np.asarray(im).transpose(2,0,1).astype(np.float32)/127.5-1 for im in images])
        arrays[prefix+'_indices'] = indices
    previous = directory/'examples.npz'
    if previous.exists():
        backup=directory/'examples.original.npz'
        if not backup.exists(): shutil.copy2(previous,backup)
        with np.load(previous) as old: best=old['best'].copy()
        np.savez_compressed(previous,images=arrays['small'],indices=arrays['small_indices'],
                            overview=arrays['overview'],overview_indices=arrays['overview_indices'],best=best)
    (directory/'panel_selection.json').write_text(json.dumps(selections,indent=2)+'\n')
    print(f'Rebuilt {directory}: {chains} ladders, {draws} draws',flush=True)
    return True


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots',nargs='+',type=Path)
    args=parser.parse_args()
    count=0
    for root in args.roots:
        for index in sorted(root.rglob('all_samples/index.csv')):
            count+=rebuild(index.parent.parent)
    print(f'Rebuilt {count} completed run panels.')
