"""Q network: Transformer encoder over the 6 anchor tokens + pairwise (primary, leaf) dueling head.

token_a = W_role(a)(slots_a) + W_pos(anchor xy) + E_role(a)          role(a) = primary if a is the state's primary
          (separate input projections for primary and leaf slots: their values mean different things)
h       = TransformerEncoder (N_LAYERS layers, FFN, residual, LayerNorm) over the 6 tokens
g       = [mean_a h_a, MLP(scalars)]
Q(p, l) = V(g) + A(p, l) - mean A,   A(p, l) = MLP([h_p, h_l, h_p * h_l, g])
"""
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 128
DROPOUT = 0.0


class DQN_Transformer(nn.Module):
    def __init__(self, n_scalar, n_onehot, n_slot, anchor_xy, action_list):
        """anchor_xy: [(x, y)] in anchor order; action_list: [[primary, leaf]] with 1-based anchor ids in that order."""
        super().__init__()
        self.n_scalar, self.n_onehot, self.n_slot = n_scalar, n_onehot, n_slot
        self.n_anchor = len(anchor_xy)
        self.register_buffer('anchor_xy', torch.tensor(anchor_xy, dtype=torch.float32))
        self.register_buffer('primary_idx', torch.tensor([p - 1 for p, _ in action_list]))
        self.register_buffer('leaf_idx', torch.tensor([l - 1 for _, l in action_list]))

        self.proj_primary = nn.Linear(n_slot, D_MODEL)
        self.proj_leaf = nn.Linear(n_slot, D_MODEL)
        self.pos_embed = nn.Sequential(nn.Linear(2, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, D_MODEL))
        self.role_embed = nn.Embedding(2, D_MODEL)                   # 0 leaf, 1 primary

        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)

        self.scalar_fc = nn.Sequential(nn.Linear(n_scalar, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        g_dim = D_MODEL + 64
        self.value = nn.Sequential(nn.Linear(g_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.pair = nn.Sequential(nn.Linear(3 * D_MODEL + g_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        B = x.shape[0]
        scalars = x[:, :self.n_scalar]
        role = scalars[:, self.n_scalar - self.n_onehot:].unsqueeze(-1)          # (B, 6, 1)
        slots = x[:, self.n_scalar:].reshape(B, self.n_anchor, self.n_slot)

        tok = role * self.proj_primary(slots) + (1 - role) * self.proj_leaf(slots)
        tok = tok + self.pos_embed(self.anchor_xy) + self.role_embed(role.squeeze(-1).long())
        h = self.encoder(tok)                                                     # (B, 6, D)

        g = torch.cat([h.mean(1), self.scalar_fc(scalars)], dim=1)               # (B, g_dim)
        hp, hl = h[:, self.primary_idx], h[:, self.leaf_idx]                      # (B, n_actions, D)
        ga = g.unsqueeze(1).expand(-1, hp.shape[1], -1)
        adv = self.pair(torch.cat([hp, hl, hp * hl, ga], dim=-1)).squeeze(-1)    # (B, n_actions)
        return self.value(g) + adv - adv.mean(dim=1, keepdim=True)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
