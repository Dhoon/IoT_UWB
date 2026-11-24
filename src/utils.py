import math

import numpy as np
from scipy.optimize import least_squares

from settings import ANCHOR_POSITIONS, ANCHORS, C, DWT_TIME_UNIT, UINT32


def calc_azimuth(x, y):
    """Azimuth in degrees, 0 = +y, clockwise, [0, 360)."""
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def azimuth_error(est_pos, true_pos):
    err = abs(calc_azimuth(*est_pos[:2]) - calc_azimuth(*true_pos[:2]))
    return 360 - err if err > 180 else err


def primary_distance(messages, primary):
    """DS-TWR distance (m) from the primary message [Ra, Da, Rb, Db, D2b]; None if absent."""
    msg = messages.get(str(primary))
    if not msg or len(msg) < 5 or any(v < 0 for v in msg[:5]):
        return None
    Ra, Da, Rb, Db = msg[:4]
    den = Ra + Rb + Da + Db
    tof = (Ra * Rb - Da * Db) / den if den != 0 else 0
    return tof * DWT_TIME_UNIT * C


def tdoa(messages, primary, leaf):
    """TDoA (m) of leaf vs primary from [t2, t5, t8] of the leaf; None if any of them is missing."""
    msg = messages.get(str(leaf))
    if not msg or len(msg) < 3:
        return None
    t2, t5, t8 = (float(v) for v in msg[:3])
    if t2 <= 0 or t5 <= 0 or t8 <= 0:
        return None
    Ra, Da, Rb, Db = messages[str(primary)][:4]
    Rt1 = (t5 - t2) % UINT32
    Rt2 = (t8 - t5) % UINT32
    den = 2 * Rt1 + 2 * Rt2
    num = Da * Rt1 - Rt2 * Ra + Rb * Rt1 - Rt2 * Db
    return (num / den if den != 0 else 0) * DWT_TIME_UNIT * C


def extract_ranging(record, primary, leaf):
    """(primary_dist, tdoa) in m, or None."""
    messages = record.get('messages', {})
    d = primary_distance(messages, primary)
    if d is None:
        return None
    t = tdoa(messages, primary, leaf)
    return None if t is None else (d, t)


def intersection_start(messages, primary, leaf, primary_dist, delta, prev_est=None):
    """Least-squares start: one of the two intersections of the primary circle (radius r1 = primary_dist) and the leaf
    circle (radius r2 = |leaf - primary| + delta), the one that fits the other leaves of the same record best:
    J(p) = sum_k (|p - a_k| - r_k)^2, r_k = |a_k - primary| + tdoa_k. Uses no ground truth and no previous estimate,
    so a mirror solution of one step does not carry over to the next.
    No intersection (|r1 - r2| > b): the two points 0.1 m off the pair line. No other leaf with a TDoA: the candidate
    nearest to prev_est, or the primary anchor position if there is none."""
    P, L = np.array(ANCHOR_POSITIONS[primary]), np.array(ANCHOR_POSITIONS[leaf])
    b = np.linalg.norm(L - P)
    e = (L - P) / b
    n = np.array([-e[1], e[0]])
    r1, r2 = primary_dist, b + delta
    a = (r1 ** 2 - r2 ** 2 + b ** 2) / (2 * b)           # along the pair line from the primary
    h2 = r1 ** 2 - a ** 2
    h = math.sqrt(h2) if h2 > 0 else 0.1
    cands = [P + a * e + h * n, P + a * e - h * n]

    others = []
    for k in ANCHORS:
        if k in (primary, leaf):
            continue
        t = tdoa(messages, primary, k)
        if t is not None:
            A = np.array(ANCHOR_POSITIONS[k])
            others.append((A, np.linalg.norm(A - P) + t))
    if others:
        return min(cands, key=lambda c: sum((np.linalg.norm(c - A) - r) ** 2 for A, r in others))
    if prev_est is not None:
        return min(cands, key=lambda c: np.linalg.norm(c - np.asarray(prev_est)))
    return P


def estimate_position_for_action(record, primary, leaf, prev_est=None):
    """(position, None) or (None, 'primary' | 'leaf'): primary circle + one TDoA hyperbola, least squares from
    intersection_start (prev_est is only its fallback when no other leaf has a TDoA)."""
    messages = record.get('messages', {})
    primary_dist = primary_distance(messages, primary)
    if primary_dist is None:
        return None, 'primary'
    delta = tdoa(messages, primary, leaf)
    if delta is None or leaf not in ANCHOR_POSITIONS:
        return None, 'leaf'

    primary_pos = np.array(ANCHOR_POSITIONS[primary])
    leaf_pos = np.array(ANCHOR_POSITIONS[leaf])
    d_leaf_primary = np.linalg.norm(leaf_pos - primary_pos)

    def residuals(p):
        return [np.linalg.norm(p - primary_pos) - primary_dist,
                np.linalg.norm(p - leaf_pos) - d_leaf_primary - delta]

    x0 = intersection_start(messages, primary, leaf, primary_dist, delta, prev_est)
    result = least_squares(residuals, x0=x0)
    return (result.x, None) if result.success else (None, 'leaf')
