import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data_loader import load_locations
from reward import get_reward
from settings import ANCHOR_DROP_PROB, ANCHORS, DATA_DIR, LOCATIONS, UINT32
from utils import calc_azimuth, estimate_position_for_action

# state = scalars (9) + 6 anchors x 4 slots (24) = 33
#   scalars: [sin az, cos az, distance_m / POS_SCALE (none -> -1), one-hot primary (6, last)]
#     az: azimuth of the previous estimate (unknown -> sin = cos = 0); distance_m of the record (root = primary)
#   primary slots: [1, Ra, Ra + Da, Ra + Da + Rb] / TS_SCALE (+ 1 except the first)
#   leaf slots: t2, t5, t8, t11 relative to the first non-zero, / TS_SCALE + 1   (t5 - t2 ~ Ra, t8 - t5 ~ Da)
#   empty slot -> -1
N_ONEHOT = 6
N_SCALAR = 3 + N_ONEHOT
N_SLOT = 4
STATE_DIM = N_SCALAR + N_SLOT * len(ANCHORS)
EMPTY = -1.0
TS_SCALE = 1e7
POS_SCALE = 10.0

class CustomEnv(gym.Env):
    """Episode = one location (cycled 1..12); step = choose (primary, leaf), draw a record of that primary."""

    def __init__(self, data_dir=DATA_DIR, locations=LOCATIONS, use_geom=True):
        super().__init__()
        data = load_locations(data_dir, locations)
        self.locations = [loc for loc in locations if loc in data]
        # records with messages, by (location, root)
        self.records = {(loc, p): [r for r in data[loc] if r.get('messages') and r.get('root_anchor') == p]
                        for loc in self.locations for p in ANCHORS}
        self.use_geom = use_geom

        self.action_list = [[p, l] for p in ANCHORS for l in ANCHORS if l != p]
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32)

        self.anchor_drop_prob = ANCHOR_DROP_PROB
        self.episode_count = 0
        self.location = self.locations[0]
        self.current_record = self.prev_record = self.prev_est = None
        self.prev_azimuth = None
        self.prev_primary = ANCHORS[0]
        self.unavailable_anchors = set()

    def _get_record(self, primary):
        if primary in self.unavailable_anchors:
            return None
        pool = self.records[(self.location, primary)]
        return random.choice(pool) if pool else None

    def _build_state(self, record, primary):
        messages = record.get('messages', {})
        slots = []
        for a in ANCHORS:
            msg = [float(v) for v in messages.get(str(a), [])]
            if a == primary:
                if len(msg) < 2 or msg[0] == 0 or msg[1] == 0:
                    slots += [EMPTY] * N_SLOT
                else:
                    Ra, Da = msg[0], msg[1]
                    Rb = msg[2] if len(msg) > 2 and msg[2] != 0 else None
                    slots += [1.0, Ra / TS_SCALE + 1, (Ra + Da) / TS_SCALE + 1,
                              EMPTY if Rb is None else (Ra + Da + Rb) / TS_SCALE + 1]
                continue
            ts = (msg + [0.0] * N_SLOT)[:N_SLOT]
            if all(v == 0 for v in ts):
                slots += [EMPTY] * N_SLOT
                continue
            base = next(v for v in ts if v != 0)
            slots += [EMPTY if v == 0 else ((v - base) % UINT32) / TS_SCALE + 1.0 for v in ts]
        dist = record.get('distance_m') if record.get('root_anchor') == primary else None
        return np.array(self._scalars(self.prev_azimuth, dist, primary) + slots, dtype=np.float32)

    @staticmethod
    def _scalars(azimuth, dist, primary):
        az = [0.0, 0.0] if azimuth is None else [math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth))]
        d = [EMPTY if dist is None else dist / POS_SCALE]
        return az + d + [1.0 if a == primary else 0.0 for a in ANCHORS]
    def reset(self, seed=None, options=None):
        fixed = (options or {}).get('location')           # test: fixed location
        while True:
            if fixed is not None:
                self.location = fixed
            else:
                self.location = self.locations[self.episode_count % len(self.locations)]
                self.episode_count += 1

            self.unavailable_anchors = {a for a in ANCHORS if random.random() < self.anchor_drop_prob}
            available = [a for a in ANCHORS if a not in self.unavailable_anchors]
            if not available:
                self.unavailable_anchors = set()
                available = list(ANCHORS)

            self.prev_primary = random.choice(available)
            self.current_record = self._get_record(self.prev_primary)
            if self.current_record is not None:
                break

        leaf = random.choice([a for a in ANCHORS if a != self.prev_primary])
        est, _ = estimate_position_for_action(self.current_record, self.prev_primary, leaf)
        self.prev_azimuth = calc_azimuth(est[0], est[1]) if est is not None else None
        self.prev_est = tuple(est) if est is not None else None
        self.prev_record = self.current_record
        return self._build_state(self.current_record, self.prev_primary), {}

    def step(self, action_idx):
        primary, leaf = self.action_list[action_idx]

        self.current_record = self._get_record(primary)
        if self.current_record is None:               # primary unavailable / no record: all its slots empty
            if self.prev_record is not None:
                state = self._build_state(self.prev_record, primary)
                state[:N_SCALAR] = self._scalars(None, None, primary)   # azimuth, distance unknown
                i = N_SCALAR + ANCHORS.index(primary) * N_SLOT
                state[i:i + N_SLOT] = EMPTY
            else:
                state = np.full(STATE_DIM, EMPTY, dtype=np.float32)
                state[:N_SCALAR] = self._scalars(None, None, primary)
            return state, -1.0, False, False, {}

        est, _ = estimate_position_for_action(self.current_record, primary, leaf, self.prev_est)
        if est is not None:
            self.prev_azimuth = calc_azimuth(est[0], est[1])
            self.prev_est = tuple(est)
        self.prev_primary = primary
        self.prev_record = self.current_record

        state = self._build_state(self.current_record, primary)
        reward, info = get_reward(self.current_record, (primary, leaf), self.use_geom)
        info['record'] = self.current_record
        info['est'] = None if est is None else [float(est[0]), float(est[1])]   # tracking estimate (prev_est start)
        return state, reward, False, False, info

    def get_action_array(self, action_idx):
        return self.action_list[action_idx]
