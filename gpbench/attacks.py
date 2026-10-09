"""Inference attacks that recover home and workplace from a (possibly protected) trace.

`infer_home_work`: DBSCAN stay points scored by when the user dwells there (night -> home,
weekday office hours -> work). `hole_attack`: finds a home hidden by a suppression zone.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from .synth import Trace


def _hours(t):
    return (t % 86400) / 3600.0


def _weekday(t):
    return (t // 86400) % 7 < 5


def stay_points(trace, eps_m=150.0, min_samples=6):
    """Cluster points; return list of dicts with center, size and dwell-hour histogram."""
    if len(trace.t) < min_samples:
        return []
    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(trace.xy)
    out = []
    for lab in sorted(set(labels) - {-1}):
        m = labels == lab
        out.append({"center": trace.xy[m].mean(axis=0), "n": int(m.sum()),
                    "hours": _hours(trace.t[m]), "weekday": _weekday(trace.t[m])})
    return out


def infer_home_work(trace, eps_m=150.0, min_samples=6):
    """Return (home_xy or None, work_xy or None)."""
    sps = stay_points(trace, eps_m, min_samples)
    if not sps:
        return None, None
    def night_frac(s): return float(np.mean((s["hours"] >= 22) | (s["hours"] < 6)))
    def night(s): return night_frac(s) * s["n"]
    def office(s): return float(np.mean((s["hours"] >= 9) & (s["hours"] < 17) & s["weekday"])) * s["n"]
    home = max(sps, key=night)
    if night_frac(home) < 0.3:       # nobody sleeps here: the home stay was hidden or never observed
        home = None
    rest = [s for s in sps if s is not home]
    work = max(rest, key=office) if rest and max(office(s) for s in rest) > 0 else None
    return (home["center"] if home else None), (work["center"] if work else None)


def hole_attack(trace, radius_m, step_m=None, gap_s=4 * 3600, sigma_perp_m=30.0):
    """Locate a home that was hidden with a suppression zone ("privacy zone").

    Suppression removes every point within `radius_m` of home, which leaves a hole in the trace
    where the user sleeps. The attacker does not need any point inside the hole:

    1. Find the long overnight gaps in the timeline.
    2. At each gap edge take the first (or last) visible point p and its neighbour along the path.
       Both lie on the straight route leaving (or entering) home, so the line through them, with
       unit direction u pointing away from home, passes through home.
    3. Home sits `radius + U(0, step)` behind p along that line: a hard constraint across the line
       (std `sigma_perp_m`, GPS noise) and a soft one along it (mean radius + step/2, std step/sqrt(12)).
    4. Combine all gaps with weighted least squares. Routes in different directions pin the
       position, repeated routes average down the along-line uncertainty.

    Assumes the attacker knows the zone radius and the sampling rate (Kerckhoffs' principle).
    Returns the estimated home xy, or None if there were no usable gaps.
    """
    t, xy = trace.t, trace.xy
    if len(t) < 4:
        return None
    if step_m is None:
        # distance covered between two samples while travelling: most steps are stays, so use a high percentile
        step_m = float(np.percentile(np.linalg.norm(np.diff(xy, axis=0), axis=1), 95))
    sigma_along = max(step_m, 1.0) / np.sqrt(12)
    ata = np.zeros((2, 2))
    atb = np.zeros(2)
    used = 0
    for g in np.where(np.diff(t) > gap_s)[0]:
        for first, second in ((g + 1, g + 2), (g, g - 1)):   # leaving home / arriving home
            if not (0 <= second < len(t)):
                continue
            d = xy[second] - xy[first]
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            u = d / n                                  # points away from home
            perp = np.array([-u[1], u[0]])
            p = xy[first]
            for vec, target, sigma in ((perp, perp @ p, sigma_perp_m),
                                       (u, u @ p - (radius_m + step_m / 2), sigma_along)):
                ata += np.outer(vec, vec) / sigma ** 2
                atb += vec * target / sigma ** 2
            used += 1
    if used == 0:
        return None
    return np.linalg.solve(ata + 1e-12 * np.eye(2), atb)
