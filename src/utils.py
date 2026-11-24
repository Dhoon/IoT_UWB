import math

import numpy as np
from scipy.optimize import least_squares

from settings import ANCHOR_POSITIONS, C, DWT_TIME_UNIT, UINT32


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


def estimate_position_for_action(record, primary, leaf, prev_est=None):
    """(position, None) or (None, 'primary' | 'leaf'): primary circle + one TDoA hyperbola, least squares."""
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

    x0 = prev_est if prev_est is not None else primary_pos   # cold start: primary anchor position
    result = least_squares(residuals, x0=x0)
    return (result.x, None) if result.success else (None, 'leaf')
