import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data_loader import load_locations
from reward import get_reward, leaf_estimates
from settings import ANCHOR_DROP_PROB, ANCHORS, DATA_DIR, LOCATIONS, UINT32
from utils import calc_azimuth, estimate_position_for_action

# state = scalars (9) + 6 anchors x 4 slots (24) = 33
#   scalars: [sin az, cos az, distance_m / POS_SCALE (none -> -1), one-hot primary (6, last)]
#     az: azimuth of the previous estimate (unknown -> sin = cos = 0); distance_m of the record (DS-TWR distance of the primary)
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

    def __init__(self, data_dir=DATA_DIR, locations=LOCATIONS, use_geom=True, all_leaves=False):
        super().__init__()
        data = load_locations(data_dir, locations)
        self.locations = [loc for loc in locations if loc in data]
        # all records, by (location, primary): failed sessions (no messages / no primary report) included -> primary fail
        self.records = {(loc, p): [r for r in data[loc] if r.get('primary_anchor') == p]
                        for loc in self.locations for p in ANCHORS}
        empty = [k for k, v in self.records.items() if not v]
        if empty:
            raise ValueError(f"no records for (location, primary) {empty} in {data_dir}")
        self.use_geom = use_geom
        self.all_leaves = all_leaves          # step() also returns the other leaves of the chosen primary (training)

        self.action_list = [[p, l] for p in ANCHORS for l in ANCHORS if l != p]
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32)

        self.anchor_drop_prob = ANCHOR_DROP_PROB
        self.episode_count = 0
        self.location = self.locations[0]
        self.current_record = self.prev_est = None
        self.prev_azimuth = None
        self.prev_primary = ANCHORS[0]
        self.unavailable_anchors = set()

    def _get_record(self, primary):
        """Random record of this primary at the current location, with the episode's anchor drop applied."""
        return self._mask_dropped(random.choice(self.records[(self.location, primary)]))

    def _mask_dropped(self, record):
        """Copy of the record with the episode's anchor drop applied.
        Dropped primary: no session -> no messages at all, no distance = same as a failed session record (primary fail).
        Dropped leaf: t2, t5, t8 removed, t11 kept -> [0, 0, 0, t11] (slots [-1, -1, -1, t11]), leaf fail (no TDoA)."""
        if not self.unavailable_anchors:
            return record
        masked = dict(record)
        if record.get('primary_anchor') in self.unavailable_anchors:
            masked['messages'], masked['distance_m'] = {}, None
            return masked
        messages = {}
        for k, v in record.get('messages', {}).items():
            if int(k) not in self.unavailable_anchors:
                messages[k] = v
            elif len(v) >= 4 and v[3] != 0:          # dropped leaf: keep t11 only
                messages[k] = [0, 0, 0, v[3]]
        masked['messages'] = messages
        return masked

    def _build_state(self, record, primary, azimuth):
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
        return np.array(self._scalars(azimuth, record.get('distance_m'), primary) + slots, dtype=np.float32)

    @staticmethod
    def _scalars(azimuth, dist, primary):
        az = [0.0, 0.0] if azimuth is None else [math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth))]
        d = [EMPTY if dist is None else dist / POS_SCALE]
        return az + d + [1.0 if a == primary else 0.0 for a in ANCHORS]
    def reset(self, seed=None, options=None):
        fixed = (options or {}).get('location')           # test: fixed location
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
        leaf = random.choice([a for a in ANCHORS if a != self.prev_primary])
        est, _ = estimate_position_for_action(self.current_record, self.prev_primary, leaf)
        self.prev_azimuth = calc_azimuth(est[0], est[1]) if est is not None else None
        self.prev_est = tuple(est) if est is not None else None
        return self._build_state(self.current_record, self.prev_primary, self.prev_azimuth), {}

    def step(self, action_idx):
        """info['other_leaves'] (all_leaves=True): [(action_idx, next_state, reward)] of the other 4 leaves of the same
        primary on the same record, as if that leaf had been chosen. They are for training only: the tracking
        (prev_est, prev_azimuth) follows the chosen action."""
        primary, leaf = self.action_list[action_idx]

        self.current_record = self._get_record(primary)
        start = self.prev_est                         # only a fallback of the solver start (utils.intersection_start)
        estimates = leaf_estimates(self.current_record, primary, start)
        reward, info = get_reward(self.current_record, (primary, leaf), estimates, self.use_geom)
        prev_az = self.prev_azimuth

        def next_az(est):                             # failed estimate -> azimuth unchanged
            return prev_az if est is None else calc_azimuth(est[0], est[1])

        if self.all_leaves:
            info['other_leaves'] = [
                (self.action_list.index([primary, l]),
                 self._build_state(self.current_record, primary, next_az(estimates[l][0])),
                 get_reward(self.current_record, (primary, l), estimates, self.use_geom)[0])
                for l in estimates if l != leaf]

        est = estimates[leaf][0]
        self.prev_azimuth = next_az(est)
        if est is not None:
            self.prev_est = tuple(est)
        self.prev_primary = primary

        state = self._build_state(self.current_record, primary, self.prev_azimuth)
        info['record'] = self.current_record
        info['est'] = None if est is None else [float(est[0]), float(est[1])]   # the estimate the reward is computed on
        return state, reward, False, False, info

    def get_action_array(self, action_idx):
        return self.action_list[action_idx]
