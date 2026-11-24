"""Reward of an action (primary, leaf) on one record.

total = R_geom + Rangle + Rbest            (use_geom=False: R_geom only logged, not added)
  R_geom : blind-zone penalty of the pair. q = (primary_dist - b - tdoa) / b (b = |leaf - primary|, q ~ cos of the
           angle between the pair line and the tag), th = 1 - GEOM_EPS / b:
           0 if |q| <= th (circles intersect with margin > GEOM_EPS), -(|q| - th) / (1 - th) if th < |q| <= 1,
           -1 if |q| > 1 (no intersection). 0 if the leaf has no TDoA.
  Rangle : azimuth error e of the estimate: 1 - e / BETA if e <= BETA, else max(-1, -(e - BETA) / BETA)
  Rbest  : -(1 - best / BETA) if the action is not within BETA but the best leaf (same primary) is, else 0
primary fail -> -1
leaf fail (no TDoA of the leaf, so no R_geom), best = error of the best leaf of the same primary:
  best <= BETA       -> Rbest = -(1 - best / BETA)                  (a good leaf was missed)
  best > BETA        -> max(-1, -(best - BETA) / BETA) = Rangle(best)  (failing is never better than the best leaf)
  no leaf works      -> -1                                          (the primary itself was a bad choice)
"""
import numpy as np

from settings import ANCHOR_POSITIONS, ANCHORS, BETA, GEOM_EPS
from utils import azimuth_error, estimate_position_for_action, extract_ranging


def calc_geom(record, primary, leaf):
    ranging = extract_ranging(record, primary, leaf)
    if ranging is None:
        return 0.0
    primary_dist, tdoa = ranging
    b = float(np.linalg.norm(np.subtract(ANCHOR_POSITIONS[leaf], ANCHOR_POSITIONS[primary])))
    q = abs((primary_dist - b - tdoa) / b)
    th = 1.0 - GEOM_EPS / b
    if q <= th:
        return 0.0
    return -min(1.0, (q - th) / (1.0 - th))


def leaf_estimates(record, primary, start=None):
    """{leaf: (pos, fail)} for every leaf of this primary on this record, same least-squares start."""
    return {leaf: estimate_position_for_action(record, primary, leaf, start) for leaf in ANCHORS if leaf != primary}


def best_leaf_error(record, estimates):
    """Smallest azimuth error over the leaf estimates of one primary (inf if none works)."""
    errs = [azimuth_error(pos, record['position']) for pos, _ in estimates.values() if pos is not None]
    return min(errs, default=float('inf'))


def get_reward(record, action, estimates, use_geom=True):
    """estimates: leaf_estimates(record, primary, start) of the env (start = previous estimate); the action's own
    estimate is estimates[leaf], the others give Rbest."""
    primary, leaf = action
    est_pos, fail = estimates[leaf]
    true_pos = record['position']
    R_geom = calc_geom(record, primary, leaf)
    g = R_geom if use_geom else 0.0
    best = None

    def out(total, case, Rangle=0.0, Rbest=0.0):
        return total, {'case': case, 'Rangle': Rangle, 'Rbest': Rbest, 'R_geom': R_geom, 'best_leaf_err': best}

    if fail == 'primary':
        return out(-1.0, 'primary_fail')

    best = best_leaf_error(record, estimates)
    Rbest_if_missed = -(1.0 - best / BETA) if best <= BETA else 0.0

    if est_pos is None:                                   # leaf fail (no TDoA of this leaf -> R_geom = 0)
        if best == float('inf'):
            return out(-1.0, 'leaf_no_best')
        if best <= BETA:                                  # a leaf within BETA was missed
            return out(Rbest_if_missed, 'leaf_fail', Rbest=Rbest_if_missed)
        r = max(-1.0, -(best - BETA) / BETA)              # no leaf within BETA: the Rangle of the best leaf (<= 0)
        return out(r, 'leaf_fail', Rbest=r)

    err = azimuth_error(est_pos, true_pos)
    if err <= BETA:
        Rangle, Rbest = 1.0 - err / BETA, 0.0
    else:
        Rangle, Rbest = max(-1.0, -(err - BETA) / BETA), Rbest_if_missed
    return out(g + Rangle + Rbest, 'success', Rangle, Rbest)
