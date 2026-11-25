"""Training of the comparison baselines (docs/IOT-D-26-03776_Training_Plan_v3.md). GA-VPL itself: train.py.

DQN baselines (T1 - T3): same env, reward, replay, epsilon-greedy, double DQN, Stage 1 -> Stage 2 (anchor drop) as
train.py; only the network (--net) and the discount (--gamma) change.
python src/train_baseline.py --run T1_dqlel  --net cnn                    --all-leaves --jobs 2
python src/train_baseline.py --run T2_mlp    --net mlp                    --all-leaves --jobs 2
python src/train_baseline.py --run T3_gamma0 --net transformer --gamma 0  --all-leaves --jobs 2
  --net cnn: model.CNNQNetwork (Adapted DQLEL), mlp: model.MLPNetwork 33-256-256-128-30, transformer: DQN_Transformer
  --gamma 0: Q target = r (myopic); the target network is then unused.

Supervised baseline (T4):
python src/train_baseline.py --run T4_supervised --supervised
  1. label table (once, shared by the seeds, runs/<run>/labels.npz + labels.csv): for every location and anchor drop
     mask (2^6 - 1, all-dropped excluded), mean reward of each of the 30 actions over ALL train records of that
     (location, primary) with the mask applied (reward.get_reward, default reward, solver start without a previous
     estimate); a*(location, mask) = argmax. The expected reward of a step depends only on the hidden location and
     the drop mask (the record is drawn after the action), so no state snapshots / resampling are needed.
  2. per seed: states of the env with uniformly random actions, SUP_EPISODES_NODROP episodes without drop +
     SUP_EPISODES_DROP with drop FINETUNE_DROP_PROB (50 steps each); label of a state = a*(location, drop mask) of
     its step (location / mask are not network inputs).
  3. per seed: model.MLPNetwork (33-256-256-128-30 logits), cross-entropy, AdamW (AMSGrad) lr LR, batch BATCH_SIZE,
     --epochs epochs, no validation: the last epoch is saved.

Seeds: --seeds, default 0-4 (baselines). Outputs as train.py: runs/NAME/models/policy_net[_ft]_seed<s>.pth (T4: only
policy_net_ft_seed<s>.pth, the final classifier), runs/NAME/log/.
"""
import argparse
import itertools
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import train
from env import N_ONEHOT, N_SCALAR, N_SLOT, STATE_DIM, CustomEnv
from model import CNNQNetwork, DQN_Transformer, MLPNetwork, Transition
from reward import get_reward, leaf_estimates
from settings import (ANCHOR_POSITIONS, ANCHORS, BATCH_SIZE, EPISODES, FINETUNE_DROP_PROB, FINETUNE_EPISODES,
                      FINETUNE_LR, LR, MAX_STEP, run_dir)
from train import Trainer, model_paths, set_seed

BASELINE_SEEDS = [0, 1, 2, 3, 4]
SUP_EPISODES_NODROP = EPISODES             # 500, as Stage 1
SUP_EPISODES_DROP = FINETUNE_EPISODES      # 300, as Stage 2


def build_net(kind, action_list, device):
    if kind == 'transformer':
        net = DQN_Transformer(N_SCALAR, N_ONEHOT, N_SLOT, [ANCHOR_POSITIONS[a] for a in ANCHORS], action_list)
    elif kind == 'cnn':
        net = CNNQNetwork(STATE_DIM, len(action_list))
    elif kind == 'mlp':
        net = MLPNetwork(STATE_DIM, len(action_list))
    else:
        raise ValueError(kind)
    return net.to(device)


# ---------------------------------------------------------------- T1 - T3: DQN baselines
class BaselineTrainer(Trainer):
    """train.Trainer with another network and discount; everything else (env, replay, epsilon, stages) unchanged."""

    def __init__(self, env, log, net, gamma):
        super().__init__(env, log)
        self.net_kind, self.gamma = net, gamma
        self.policy_net = build_net(net, env.action_list, self.device)
        self.target_net = build_net(net, env.action_list, self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_params, self.target_params = list(self.policy_net.parameters()), list(self.target_net.parameters())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)

    def optimize(self):
        """train.Trainer.optimize with self.gamma (gamma = 0: target = reward, no next-state evaluation)."""
        if len(self.memory) < BATCH_SIZE:
            return
        batch = Transition(*zip(*self.memory.sample(BATCH_SIZE)))
        states, actions, rewards = torch.cat(batch.state), torch.cat(batch.action), torch.cat(batch.reward)
        q = self.policy_net(states).gather(1, actions)
        target = rewards
        if self.gamma > 0:
            non_final = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
            next_states = torch.cat([s for s in batch.next_state if s is not None])
            next_v = torch.zeros(BATCH_SIZE, device=self.device)
            with torch.no_grad():                                # double DQN
                next_a = self.policy_net(next_states).argmax(1, keepdim=True)
                next_v[non_final] = self.target_net(next_states).gather(1, next_a).squeeze(1)
            target = rewards + self.gamma * next_v
        loss = nn.SmoothL1Loss()(q, target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


def open_log(run, seed):
    stamp = f"train_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = run_dir(run, 'log')
    f = open(os.path.join(log_dir, f"{stamp}.txt"), 'w', encoding='utf-8')

    def log(msg):
        print(f"s{seed} {msg}")
        f.write(msg + '\n')
        f.flush()
    return log, f, os.path.join(log_dir, stamp)


def train_dqn_seed(run, seed, net, gamma, all_leaves):
    """As train.train_seed (default reward, R_geom on), with BaselineTrainer."""
    set_seed(seed)
    log, log_file, stamp_path = open_log(run, seed)
    env = CustomEnv(all_leaves=all_leaves)
    env.action_space.seed(seed)
    tr = BaselineTrainer(env, log, net, gamma)
    tr.seed = seed
    s1, s2 = model_paths(run, '', seed)
    n_params = sum(p.numel() for p in tr.policy_net.parameters())
    log(f"run={run} seed={seed} device={tr.device} net={net} params={n_params} gamma={gamma} "
        f"all_leaves={all_leaves} use_best={env.use_best} memory={tr.memory_size}")
    if os.path.exists(s1):
        tr.load(s1)
        log(f"Stage 1 skipped. Loaded {os.path.basename(s1)}")
    else:
        log(f"Stage 1 | seed={seed} episodes={EPISODES} drop={env.anchor_drop_prob} lr={LR}")
        tr.run(EPISODES, 'S1', s1)
    tr.start_finetune()
    log(f"Stage 2 | seed={seed} episodes={FINETUNE_EPISODES} drop={FINETUNE_DROP_PROB} lr={FINETUNE_LR}")
    tr.run(FINETUNE_EPISODES, 'S2', s2, offset=EPISODES)
    pd.DataFrame(tr.history).to_csv(f"{stamp_path}_episodes.csv", index=False, float_format='%.4f')
    log_file.close()
    env.close()


def train_dqn_worker(*args):
    torch.set_num_threads(1)
    train_dqn_seed(*args)


# ---------------------------------------------------------------- T4: supervised
ALL_MASKS = [frozenset(m) for k in range(len(ANCHORS)) for m in itertools.combinations(ANCHORS, k)]   # 63, no all-dropped


def label_rows_for_location(loc):
    """Mean reward of the 30 actions for every drop mask at one location: [(loc, mask, means (30,), n_records (30,))]."""
    env = CustomEnv(locations=[loc])
    out = []
    for mask in ALL_MASKS:
        env.unavailable_anchors = set(mask)
        sums, counts = np.zeros(len(env.action_list)), np.zeros(len(env.action_list))
        for p in ANCHORS:
            for rec in env.records[(loc, p)]:
                r = env._mask_dropped(rec)
                est = leaf_estimates(r, p)
                for l in est:
                    i = env.action_list.index([p, l])
                    sums[i] += get_reward(r, (p, l), est, env.use_geom, env.use_best)[0]
                    counts[i] += 1
        out.append((loc, mask, sums / counts, counts))
    env.close()
    return out


def mask_key(mask):
    return ''.join('1' if a in mask else '0' for a in ANCHORS)     # anchor 1..6 dropped?


def label_table(run, log, workers):
    """{(location, mask_key): a*} and the full table, computed once and cached in runs/<run>/labels.npz."""
    path = os.path.join(run_dir(run, 'log'), '..', 'labels.npz')
    path = os.path.normpath(path)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        log(f"label table loaded from {path}")
        return {(int(l), str(m)): int(a) for l, m, a in zip(z['location'], z['mask'], z['label'])}
    locs = CustomEnv().locations
    log(f"label table: {len(locs)} locations x {len(ALL_MASKS)} masks x 30 actions, workers={workers}")
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
        rows = [r for part in ex.map(label_rows_for_location, locs) for r in part]
    loc_a = np.array([r[0] for r in rows])
    mask_a = np.array([mask_key(r[1]) for r in rows])
    means = np.stack([r[2] for r in rows])
    counts = np.stack([r[3] for r in rows])
    labels = means.argmax(1)
    np.savez(path, location=loc_a, mask=mask_a, mean_reward=means, n_records=counts, label=labels)
    actions = CustomEnv(locations=[locs[0]]).action_list
    pd.DataFrame({'location': loc_a, 'drop_mask(anchors 1-6)': mask_a, 'label': labels,
                  'pair': [tuple(actions[a]) for a in labels], 'best_mean_reward': means.max(1)}
                 ).to_csv(path.replace('.npz', '.csv'), index=False, float_format='%.4f')
    log(f"label table saved: {path}")
    return {(int(l), str(m)): int(a) for l, m, a in zip(loc_a, mask_a, labels)}


def collect_states(seed, labels, log):
    """States of the env under uniformly random actions; label = a*(location, drop mask) of the step."""
    set_seed(seed)
    env = CustomEnv()
    env.action_space.seed(seed)
    X, y, meta = [], [], []
    for n_ep, drop in ((SUP_EPISODES_NODROP, 0.0), (SUP_EPISODES_DROP, FINETUNE_DROP_PROB)):
        env.anchor_drop_prob = drop
        for _ in range(n_ep):
            obs, _ = env.reset()
            key = (env.location, mask_key(env.unavailable_anchors))
            for _ in range(MAX_STEP):
                X.append(obs)
                y.append(labels[key])
                meta.append((env.location, key[1], drop))
                obs, *_ = env.step(env.action_space.sample())
    env.close()
    log(f"states: {len(X)} (episodes {SUP_EPISODES_NODROP} without drop + {SUP_EPISODES_DROP} with drop "
        f"{FINETUNE_DROP_PROB}), distinct labels {len(set(y))}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), meta


def train_supervised_seed(run, seed, labels, epochs):
    log, log_file, stamp_path = open_log(run, seed)
    X, y, meta = collect_states(seed, labels, log)
    pd.DataFrame(meta, columns=['location', 'drop_mask', 'drop_prob']).assign(label=y).to_csv(
        f"{stamp_path}_states_meta.csv", index=False)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions = CustomEnv(locations=[1]).action_list
    net = build_net('mlp', actions, device)
    opt = optim.AdamW(net.parameters(), lr=LR, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss()                      # takes raw logits
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    log(f"run={run} seed={seed} device={device} net=mlp params={sum(p.numel() for p in net.parameters())} "
        f"epochs={epochs} lr={LR} batch={BATCH_SIZE}")
    hist = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=device)
        tot, correct = 0.0, 0
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            logits = net(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            correct += (logits.argmax(1) == yt[idx]).sum().item()
        hist.append(dict(epoch=ep + 1, loss=tot / len(Xt), train_acc=correct / len(Xt)))
        log(f"[SUP] epoch {ep + 1:3d}/{epochs} loss={tot / len(Xt):.4f} train_acc={correct / len(Xt):.3f}")
    s2 = model_paths(run, '', seed)[1]
    torch.save(net.state_dict(), s2)
    log(f"saved {os.path.basename(s2)} (last epoch)")
    pd.DataFrame(hist).to_csv(f"{stamp_path}_epochs.csv", index=False, float_format='%.4f')
    log_file.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--net', choices=['cnn', 'mlp', 'transformer'], default='mlp')
    ap.add_argument('--gamma', type=float, default=train.GAMMA, help='discount (T3: 0)')
    ap.add_argument('--all-leaves', action='store_true')
    ap.add_argument('--supervised', action='store_true', help='T4: label table + random-policy states + CE')
    ap.add_argument('--epochs', type=int, default=50, help='--supervised: training epochs (last one is saved)')
    ap.add_argument('--seeds', type=int, nargs='+', default=BASELINE_SEEDS)
    ap.add_argument('--jobs', type=int, default=1, help='DQN: seeds in parallel; --supervised: label-table workers')
    args = ap.parse_args()

    if args.jobs > 1:
        for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[v] = '1'

    if args.supervised:
        os.makedirs(run_dir(args.run, 'log'), exist_ok=True)
        labels = label_table(args.run, print, max(args.jobs, 1))
        for s in args.seeds:
            train_supervised_seed(args.run, s, labels, args.epochs)
        return

    jobs = [(args.run, s, args.net, args.gamma, args.all_leaves) for s in args.seeds]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context('spawn')) as ex:
            for f in [ex.submit(train_dqn_worker, *j) for j in jobs]:
                f.result()
    else:
        for j in jobs:
            train_dqn_seed(*j)


if __name__ == "__main__":
    main()
