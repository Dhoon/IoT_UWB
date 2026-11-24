"""Training curve over seeds: Stage 1 and Stage 2 on one continuous episode axis.

Reads log/train[_wo_rgeom]_seed<s>_<YYYYmmdd_HHMMSS>_episodes.csv (train.py; latest run per seed).
Per seed: moving average over --window episodes (default 12 = one cycle of locations 1..12).
Across seeds: mean line, +-1 std band, each seed as a thin line. Dashed line = start of Stage 2 (fine-tuning).
Panels: (a) episode reward, (b) success rate = success steps / steps with a reward case.
Variants: full reward (blue), w/o R_geom (orange), each drawn if its logs exist.

python src/plot_training_curve.py [--seeds 0 1 2 3 4] [--window 12]
Outputs: result/training_curve_<YYYYmmdd_HHMMSS>.png / .pdf / .csv (plotted mean, std, n_seeds per episode)
"""
import argparse
import glob
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from settings import ROOT, SEEDS

VARIANTS = [('', 'Full reward', '#2a78d6'), ('_wo_rgeom', 'w/o $R_{geom}$', '#eb6834')]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
PANELS = [('reward', 'Episode reward'), ('success_rate', 'Success rate')]


def load_runs(tag, seeds):
    """{seed: episodes DataFrame} of the latest run of each seed."""
    pat = re.compile(rf"train{re.escape(tag)}_seed(\d+)_(\d{{8}}_\d{{6}})_episodes\.csv$")
    latest = {}
    for f in glob.glob(os.path.join(ROOT, 'log', '*_episodes.csv')):
        m = pat.search(os.path.basename(f))
        if m and int(m.group(1)) in seeds and m.group(2) > latest.get(int(m.group(1)), ('', ''))[0]:
            latest[int(m.group(1))] = (m.group(2), f)
    runs = {}
    for s, (_, f) in sorted(latest.items()):
        d = pd.read_csv(f).sort_values('episode')
        d['success_rate'] = d.n_success / d.n_steps_with_case.where(d.n_steps_with_case > 0)
        runs[s] = d
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--window', type=int, default=12, help='moving-average window in episodes')
    args = ap.parse_args()

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(7, 6), sharex=True)
    rows, drawn, s2_start = [], 0, None
    for tag, label, color in VARIANTS:
        runs = load_runs(tag, args.seeds)
        if not runs:
            continue
        drawn += 1
        s2 = [d.episode[d.stage == 'S2'].min() for d in runs.values() if (d.stage == 'S2').any()]
        s2_start = min(s2) if s2 else s2_start
        for (col, _), ax in zip(PANELS, axes):
            sm = pd.DataFrame({s: d.set_index('episode')[col].rolling(args.window, min_periods=1).mean()
                               for s, d in runs.items()})
            for s in sm:
                ax.plot(sm.index, sm[s], color=color, lw=0.6, alpha=0.3)
            mean, std = sm.mean(axis=1), sm.std(axis=1)
            ax.fill_between(sm.index, mean - std, mean + std, color=color, alpha=0.18, lw=0)
            ax.plot(sm.index, mean, color=color, lw=2, label=f"{label} (n = {len(runs)} seeds)")
            rows.append(pd.DataFrame(dict(variant=label, metric=col, episode=sm.index, mean=mean.values,
                                          std=std.values, n_seeds=sm.notna().sum(axis=1).values)))
    if not drawn:
        print('no *_episodes.csv in log/')
        return

    for (col, ylabel), ax, letter in zip(PANELS, axes, 'ab'):
        if s2_start is not None:
            ax.axvline(s2_start - 0.5, color=INK2, ls='--', lw=1)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter}) {ylabel}", loc='left', fontsize=10, color=INK)
        if s2_start is not None:
            top = ax.get_ylim()[1]
            ax.text(s2_start - 0.5, top, 'Stage 1  ', ha='right', va='top', color=INK2, fontsize=9)
            ax.text(s2_start - 0.5, top, '  Stage 2 (fine-tuning)', ha='left', va='top', color=INK2, fontsize=9)
    axes[-1].set_xlabel(f"Episode (moving average over {args.window} episodes; band = ±1 std across seeds)")
    axes[0].legend(frameon=False, loc='lower right', fontsize=9)
    fig.tight_layout()

    stamp = f"training_curve_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(os.path.join(ROOT, 'result'), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ROOT, 'result', f"{stamp}.{ext}"), dpi=200)
    pd.concat(rows).to_csv(os.path.join(ROOT, 'result', f"{stamp}.csv"), index=False, float_format='%.4f')
    print(f"saved result/{stamp}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
