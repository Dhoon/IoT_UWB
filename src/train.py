"""DQN (Transformer, pairwise head) training, 2 stages: Stage 1 (no anchor drop) -> Stage 2 fine-tuning (anchor drop).

python src/train.py --run NAME                  reward = R_geom + Rangle + Rbest, seeds 0-4 (settings.SEEDS)
python src/train.py --run NAME --no-geom        ablation: R_geom not added (only logged); outputs *_wo_rgeom
python src/train.py --run NAME --seeds 0 1 2    chosen seeds
python src/train.py --run NAME --all-leaves     each step also stores the other 4 leaves of the chosen primary (same
                                                record and start, reward and next-state azimuth of that leaf) as
                                                transitions; replay memory 15000 (MEMORY_SIZE_ALL_LEAVES)
--run (required): outputs go to runs/NAME/ so that different runs never share model files.
Each seed is an independent run (seeded python / numpy / torch / action sampling, fresh network and replay memory).
Outputs per seed: runs/NAME/models/policy_net[_ft][_wo_rgeom]_seed<s>.pth (best of each stage),
                  runs/NAME/log/train[_wo_rgeom]_seed<s>_<time>.txt and _episodes.csv (one row per episode).
Stage 1 of a seed is skipped if its model file already exists in runs/NAME/models (resume within the same run).
"""
import argparse
import math
import os
import random
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from env import N_ONEHOT, N_SCALAR, N_SLOT, CustomEnv
from model import DQN_Transformer, ReplayMemory, Transition
from settings import (ANCHOR_POSITIONS, ANCHORS, BATCH_SIZE, EPISODES, EPS_DECAY, EPS_END, EPS_START, FINETUNE_DROP_PROB,
                      FINETUNE_EPISODES, FINETUNE_EPS_START, FINETUNE_LR, GAMMA, LR, MAX_STEP, MEMORY_SIZE,
                      MEMORY_SIZE_ALL_LEAVES, SEEDS, TAU, run_dir)

REWARD_KEYS = ('Rangle', 'Rbest', 'R_geom')


class Trainer:
    def __init__(self, env, log):
        self.env, self.log = env, log
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def net():
            return DQN_Transformer(N_SCALAR, N_ONEHOT, N_SLOT, [ANCHOR_POSITIONS[a] for a in ANCHORS],
                                   env.action_list).to(self.device)
        self.policy_net, self.target_net = net(), net()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory_size = MEMORY_SIZE_ALL_LEAVES if env.all_leaves else MEMORY_SIZE
        self.memory = ReplayMemory(self.memory_size)
        self.steps_done = 0
        self.eps_start = EPS_START
        self.episode_rewards = []
        self.history = []          # one row per episode (saved as CSV per seed)
        self.seed = None

    # ---------- model files ----------
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def start_finetune(self):
        self.steps_done = 0
        self.eps_start = FINETUNE_EPS_START
        self.env.anchor_drop_prob = FINETUNE_DROP_PROB
        self.memory = ReplayMemory(self.memory_size)
        for g in self.optimizer.param_groups:
            g['lr'] = FINETUNE_LR

    # ---------- action ----------
    def select_action(self, state):
        eps = EPS_END + (self.eps_start - EPS_END) * math.exp(-self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    # ---------- learning ----------
    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = Transition(*zip(*self.memory.sample(BATCH_SIZE)))
        non_final = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
        next_states = torch.cat([s for s in batch.next_state if s is not None])
        states, actions, rewards = torch.cat(batch.state), torch.cat(batch.action), torch.cat(batch.reward)

        q = self.policy_net(states).gather(1, actions)
        next_v = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():                                    # double DQN
            next_a = self.policy_net(next_states).argmax(1, keepdim=True)
            next_v[non_final] = self.target_net(next_states).gather(1, next_a).squeeze(1)
        target = rewards + GAMMA * next_v

        loss = nn.SmoothL1Loss()(q, target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update(self):
        t, p = self.target_net.state_dict(), self.policy_net.state_dict()
        for k in p:
            t[k] = p[k] * TAU + t[k] * (1 - TAU)
        self.target_net.load_state_dict(t)

    # ---------- episodes ----------
    def run(self, n_episodes, stage, save_path, offset=0):
        """Train n_episodes; save the model whenever the mean reward of the last cycle of locations (one episode per
        location) is the best so far in this stage. Ends with the best model loaded."""
        window = len(self.env.locations)
        stage_rewards, best = [], -float('inf')
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            total, cases, sums, n_succ, n_all = 0.0, {}, dict.fromkeys(REWARD_KEYS, 0.0), 0, 0

            for _ in range(MAX_STEP):
                az, prev_p = self.env.prev_azimuth, self.env.prev_primary
                action = self.select_action(state)
                obs, reward, _, _, info = self.env.step(action.item())
                self.log(f"[STEP] az={az if az is None else round(az, 2)} prev_primary={prev_p} action={self.env.get_action_array(action.item())} "
                         f"reward={reward:+.3f}")
                total += reward

                cases[info['case']] = cases.get(info['case'], 0) + 1
                n_all += 1
                if info['case'] == 'success':
                    n_succ += 1
                    for k in REWARD_KEYS:
                        sums[k] += info[k]

                next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.memory.push(state, action, next_state, torch.tensor([reward], device=self.device))
                for a, s, r in info.get('other_leaves', ()):            # --all-leaves: same primary, other leaves
                    self.memory.push(state, torch.tensor([[a]], device=self.device, dtype=torch.long),
                                     torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0),
                                     torch.tensor([r], device=self.device))
                state = next_state
                self.optimize()
                self.soft_update()

            self.episode_rewards.append(total)
            ns = max(n_succ, 1)
            self.log(f"[{stage}] Ep {offset + ep + 1:4d}/{offset + n_episodes} | location={self.env.location} "
                     f"| reward={total:+7.2f} | success={n_succ}/{n_all}")
            self.log("  cases : " + '  '.join(f"{k}={v}" for k, v in sorted(cases.items())))
            self.log("  avg   : " + '  '.join(f"{k}={sums[k] / ns:+.3f}" for k in REWARD_KEYS) + "  (success steps)")
            self.log("  total : " + '  '.join(f"{k}={sums[k]:+.1f}" for k in REWARD_KEYS))
            self.plot()

            self.history.append(dict(stage=stage, episode=offset + ep + 1, location=self.env.location, reward=total,
                                     n_success=n_succ, n_steps_with_case=n_all,
                                     **{f"{k}_sum": sums[k] for k in REWARD_KEYS}))
            stage_rewards.append(total)
            if len(stage_rewards) >= window or ep == n_episodes - 1:
                score = sum(stage_rewards[-window:]) / len(stage_rewards[-window:])
                if score > best:
                    best = score
                    self.save(save_path)
                    self.log(f"  [BEST] mean reward of last {min(window, len(stage_rewards))} episodes {score:+.2f} "
                             f"-> saved {os.path.basename(save_path)}")
        self.load(save_path)
        self.log(f"{stage} best mean reward {best:+.2f} (loaded {os.path.basename(save_path)})")

    def plot(self):
        plt.figure(1)
        plt.clf()
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'DQN Training Progress (seed {self.seed})' + ('' if self.env.use_geom else ' (w/o R_geom)'))
        plt.plot(self.episode_rewards, label='Episode Reward')
        if len(self.episode_rewards) >= 10:
            r = torch.tensor(self.episode_rewards, dtype=torch.float)
            ma = torch.cat((torch.zeros(9), r.unfold(0, 10, 1).mean(1)))
            plt.plot(ma.numpy(), label='Moving Avg (10)')
        plt.legend()
        plt.pause(0.001)
        if 'inline' in matplotlib.get_backend():
            from IPython import display
            display.display(plt.gcf())
            display.clear_output(wait=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_paths(run, tag, seed):
    """(Stage 1, Stage 2) model files of one seed in runs/<run>/models."""
    return tuple(os.path.join(run_dir(run, 'models'), f"policy_net{x}{tag}_seed{seed}.pth") for x in ('', '_ft'))


def train_seed(run, seed, use_geom, tag, all_leaves=False):
    """One independent run: seeds python / numpy / torch / action sampling, fresh env, network, replay memory."""
    set_seed(seed)
    stamp = f"train{tag}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = run_dir(run, 'log')
    log_file = open(os.path.join(log_dir, f"{stamp}.txt"), 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    env = CustomEnv(use_geom=use_geom, all_leaves=all_leaves)
    env.action_space.seed(seed)
    tr = Trainer(env, log)
    tr.seed = seed
    s1, s2 = model_paths(run, tag, seed)
    log(f"run={run} seed={seed} device={tr.device} all_leaves={all_leaves} memory={tr.memory_size}")

    if os.path.exists(s1):
        tr.load(s1)
        log(f"Stage 1 skipped. Loaded {os.path.basename(s1)}")
    else:
        log('=' * 60)
        log(f"Stage 1 | seed={seed} episodes={EPISODES} drop={env.anchor_drop_prob} lr={LR} locations={env.locations}")
        log('=' * 60)
        tr.run(EPISODES, 'S1', s1)
        log(f"Stage 1 complete. Best model in {os.path.basename(s1)}")

    tr.start_finetune()
    log('=' * 60)
    log(f"Stage 2 | seed={seed} episodes={FINETUNE_EPISODES} drop={FINETUNE_DROP_PROB} lr={FINETUNE_LR}")
    log('=' * 60)
    tr.run(FINETUNE_EPISODES, 'S2', s2, offset=EPISODES)
    log(f"Stage 2 complete. Best model in {os.path.basename(s2)}")

    pd.DataFrame(tr.history).to_csv(os.path.join(log_dir, f"{stamp}_episodes.csv"), index=False, float_format='%.4f')
    log_file.close()
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run name: outputs in runs/<run>/{models, log}')
    ap.add_argument('--no-geom', action='store_true', help='ablation: R_geom not added to the reward')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS, help=f'training seeds (default {SEEDS})')
    ap.add_argument('--all-leaves', action='store_true',
                    help='also train on the other 4 leaves of the chosen primary (same record), memory 15000')
    args = ap.parse_args()
    tag = '_wo_rgeom' if args.no_geom else ''
    plt.ion()
    for seed in args.seeds:
        train_seed(args.run, seed, not args.no_geom, tag, args.all_leaves)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()