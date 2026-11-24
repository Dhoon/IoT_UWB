"""Test of the trained policies (one per training seed) on the test split (<TEST_DIR>/<location>.json, last 20 %).

Per model and location 1..12: one episode, greedy policy (no exploration, same invalid-primary mask as training), no
anchor drop, steps = number of test records with messages at that location. Each step: the chosen primary draws a test
record of that root (as in training) and the tracking estimate starts from the previous estimate. The record draws use
--eval-seed, the same for every model.

python src/test.py                       models/policy_net_ft_seed<s>.pth, s in settings.SEEDS
python src/test.py --no-geom             models/policy_net_ft_wo_rgeom_seed<s>.pth
python src/test.py --seeds 0 1 2         chosen training seeds
python src/test.py --model X.pth         one model file
Outputs (result/test[_wo_rgeom]_<time>_*.csv, log/test[_wo_rgeom]_<time>.txt):
  _steps.csv        one row per step (column seed = training seed)
  _summary.csv      per (seed, location) and per seed over all locations (location = all)
  _across_seeds.csv per location and all: mean, std (ddof 1), min, max over the seeds of each _summary column
"""
import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from env import CustomEnv
from reward import best_leaf_error
from settings import ANCHORS, ROOT, SEEDS, TEST_DIR
from train import Trainer, model_paths, set_seed
from utils import azimuth_error, calc_azimuth

CASES = ('success', 'leaf_fail', 'leaf_no_best', 'primary_fail', 'no_record')


def evaluate(model, use_geom, eval_seed, log):
    """Step rows of one model over all test locations."""
    set_seed(eval_seed)
    env = CustomEnv(data_dir=TEST_DIR, use_geom=use_geom)
    tr = Trainer(env, log)
    tr.load(model)
    tr.policy_net.eval()
    dev = tr.device
    rows = []
    for loc in env.locations:
        n_steps = sum(len(env.records[(loc, p)]) for p in ANCHORS)
        obs, _ = env.reset(options={'location': loc})
        state = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        for t in range(n_steps):
            t0 = time.perf_counter()
            with torch.no_grad():
                a = tr.mask_invalid(tr.policy_net(state), state).max(1).indices.item()
            latency = (time.perf_counter() - t0) * 1000
            primary, leaf = env.get_action_array(a)
            obs, reward, _, _, info = env.step(a)
            rec = info.get('record')
            row = dict(location=loc, step=t, primary=primary, leaf=leaf, case=info.get('case', 'no_record'),
                       reward=reward, Rangle=info.get('Rangle'), Rbest=info.get('Rbest'), R_geom=info.get('R_geom'),
                       seq=rec['seq'] if rec else None, timestamp=rec['timestamp'] if rec else None, latency_ms=latency)
            if rec and rec.get('position') is not None:
                true = rec['position']
                row.update(true_x=true[0], true_y=true[1], true_az=calc_azimuth(*true[:2]))
                if info.get('est') is not None:
                    est = info['est']
                    row.update(est_x=est[0], est_y=est[1], est_az=calc_azimuth(*est), az_err=azimuth_error(est, true),
                               pos_err=float(np.hypot(est[0] - true[0], est[1] - true[1])))
                if row['case'] in ('success', 'leaf_fail', 'leaf_no_best'):
                    b = best_leaf_error(rec, primary, true)
                    row['best_leaf_az_err'] = b if np.isfinite(b) else None
            rows.append(row)
            state = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
    env.close()
    return rows


def summarize(g):
    s = dict(steps=len(g), reward_sum=g.reward.sum(), reward_mean=g.reward.mean())
    for c in CASES:
        s[f"n_{c}"] = int((g.case == c).sum())
    e = g.az_err.dropna() if 'az_err' in g else pd.Series(dtype=float)
    p = g.pos_err.dropna() if 'pos_err' in g else pd.Series(dtype=float)
    s.update(az_err_median=e.median(), az_err_mean=e.mean(), az_err_p90=e.quantile(.9),
             az_within_5deg=(e <= 5).mean(), pos_err_median=p.median())
    for a in ANCHORS:
        s[f"primary{a}_share"] = (g.primary == a).mean()
    s['latency_ms_mean'] = g.latency_ms.mean()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-geom', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS, help='training seeds whose models are tested')
    ap.add_argument('--model', default=None, help='test this one model file instead (seed column = -1)')
    ap.add_argument('--eval-seed', type=int, default=0, help='seed of the test record draws (same for all models)')
    args = ap.parse_args()
    tag = '_wo_rgeom' if args.no_geom else ''
    models = [(-1, args.model)] if args.model else [(s, model_paths(tag, s)[1]) for s in args.seeds]

    stamp = f"test{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    for d in ('log', 'result'):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    log_file = open(os.path.join(ROOT, 'log', f"{stamp}.txt"), 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    rows = []
    for seed, model in models:
        if not os.path.exists(model):
            log(f"[SKIP] {model} not found")
            continue
        log(f"model={os.path.basename(model)} seed={seed} data={TEST_DIR} eval_seed={args.eval_seed}")
        rows += [dict(seed=seed, **r) for r in evaluate(model, not args.no_geom, args.eval_seed, log)]
    if not rows:
        log_file.close()
        return
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(ROOT, 'result', f"{stamp}_steps.csv"), index=False, float_format='%.4f')

    summ = []
    for seed, gs in S.groupby('seed'):
        for loc, g in gs.groupby('location'):
            summ.append(dict(seed=seed, location=str(loc), **summarize(g)))
        summ.append(dict(seed=seed, location='all', **summarize(gs)))
    M = pd.DataFrame(summ)
    M.to_csv(os.path.join(ROOT, 'result', f"{stamp}_summary.csv"), index=False, float_format='%.4f')

    cols = [c for c in M.columns if c not in ('seed', 'location')]
    A = M.groupby('location', sort=False)[cols].agg(['mean', 'std', 'min', 'max'])
    A.columns = [f"{c}_{s}" for c, s in A.columns]
    A.insert(0, 'n_seeds', M.groupby('location', sort=False).seed.nunique())
    A.reset_index().to_csv(os.path.join(ROOT, 'result', f"{stamp}_across_seeds.csv"), index=False, float_format='%.4f')

    show = ['reward_mean', 'az_err_median', 'az_within_5deg', 'pos_err_median']
    log(M[['seed', 'location'] + show].round(3).to_string(index=False))
    log('\nacross seeds (mean ± std):')
    for loc, r in A.iterrows():
        log(f"  location {loc:>3} | " + '  '.join(f"{c}={r[c + '_mean']:.3f}±{r[c + '_std']:.3f}" for c in show))
    log_file.close()


if __name__ == "__main__":
    main()