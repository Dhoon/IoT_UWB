"""Reward of an action (primary, leaf) on one record.

total = R_geom + Rangle + Rbest            (use_geom=False: R_geom only logged, not added)
  R_geom : TDoA geometry quality (primary_dist - d_primary_leaf - tdoa) / d_primary_leaf, clipped to [-1, 1]
  Rangle : azimuth error e of the estimate: 1 - e / BETA if e <= BETA, else max(-1, -(e - BETA) / BETA)
  Rbest  : -(1 - best / BETA) if the action is not within BETA but the best leaf (same primary) is, else 0
primary fail -> -1;  leaf fail -> R_geom + Rbest (no other leaf works either: R_geom - 1)
"""
import numpy as np

from settings import ANCHOR_POSITIONS, ANCHORS, BETA
from utils import azimuth_error, estimate_position_for_action, extract_ranging


def calc_geom(record, primary, leaf):
    ranging = extract_ranging(record, primary, leaf)
    if ranging is None:
        return 0.0
    primary_dist, tdoa = ranging
    d_primary_leaf = float(np.linalg.norm(np.subtract(ANCHOR_POSITIONS[leaf], ANCHOR_POSITIONS[primary])))
    if d_primary_leaf == 0:
        return 0.0
    return float(np.clip((primary_dist - d_primary_leaf - tdoa) / d_primary_leaf, -1.0, 1.0))


def best_leaf_error(record, primary, true_pos, start=None):
    """Smallest azimuth error over all leaves of this primary (inf if none works), same least-squares start."""
    best = float('inf')
    for leaf in ANCHORS:
        if leaf == primary:
            continue
        pos, _ = estimate_position_for_action(record, primary, leaf, start)
        if pos is not None:
            best = min(best, azimuth_error(pos, true_pos))
    return best


def get_reward(record, action, est_pos, fail, start, use_geom=True):
    """est_pos, fail: the env's estimate of this action (estimate_position_for_action from start = previous estimate);
    start: that least-squares start, also used for the other leaves in Rbest."""
    primary, leaf = action
    true_pos = record['position']
    R_geom = calc_geom(record, primary, leaf)
    g = R_geom if use_geom else 0.0
    best = None

    def out(total, case, Rangle=0.0, Rbest=0.0):
        return total, {'case': case, 'Rangle': Rangle, 'Rbest': Rbest, 'R_geom': R_geom, 'best_leaf_err': best}

    if fail == 'primary':
        return out(-1.0, 'primary_fail')

    best = best_leaf_error(record, primary, true_pos, start)
    Rbest_if_missed = -(1.0 - best / BETA) if best <= BETA else 0.0

    if est_pos is None:                                   # leaf fail
        if best == float('inf'):
            return out(g - 1.0, 'leaf_no_best')
        return out(g + Rbest_if_missed, 'leaf_fail', Rbest=Rbest_if_missed)

    err = azimuth_error(est_pos, true_pos)
    if err <= BETA:
        Rangle, Rbest = 1.0 - err / BETA, 0.0
    else:
        Rangle, Rbest = max(-1.0, -(err - BETA) / BETA), Rbest_if_missed
    return out(g + Rangle + Rbest, 'success', Rangle, Rbest)
