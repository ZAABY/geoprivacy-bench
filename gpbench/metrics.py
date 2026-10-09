"""Privacy and utility metrics."""
from __future__ import annotations

import numpy as np

from .attacks import infer_home_work, stay_points


def privacy_scores(trace, true_home, true_work, success_radius_m=200.0):
    """Attack outcome for one user: error in meters (inf if the attack found nothing) and hit flags."""
    home, work = infer_home_work(trace)
    eh = float(np.linalg.norm(home - true_home)) if home is not None else np.inf
    ew = float(np.linalg.norm(work - true_work)) if work is not None else np.inf
    return {"home_err": eh, "work_err": ew, "home_hit": eh <= success_radius_m, "work_hit": ew <= success_radius_m}


def path_length(trace):
    return float(np.linalg.norm(np.diff(trace.xy, axis=0), axis=1).sum()) if len(trace.t) > 1 else 0.0


def utility_scores(original, protected, orig_sp=None):
    """Utility of the protected trace relative to the original.

    - length_ratio: protected path length / original (noise inflates it, thinning shrinks it)
    - stay_recall: share of the original's stay points with a protected stay point within 250 m
    - mean_displacement: mean distance between each retained point and its original position
    """
    orig_sp = stay_points(original) if orig_sp is None else orig_sp  # callers can pass a cached result
    prot_sp = stay_points(protected)
    if orig_sp:
        hits = sum(any(np.linalg.norm(o["center"] - p["center"]) <= 250 for p in prot_sp) for o in orig_sp)
        recall = hits / len(orig_sp)
    else:
        recall = float("nan")
    idx = np.clip(np.searchsorted(original.t, protected.t), 0, len(original.t) - 1)
    disp = float(np.mean(np.linalg.norm(original.xy[idx] - protected.xy, axis=1))) if len(protected.t) else float("nan")
    lo = path_length(original)
    return {"length_ratio": path_length(protected) / lo if lo else float("nan"),
            "stay_recall": recall, "mean_displacement": disp}


def uniqueness(traces, p_points, cell_m, n_trials=200, seed=0):
    """Share of users re-identified by `p_points` known (place, hour) observations.

    In the spirit of de Montjoye et al. (2013, "Unique in the Crowd"). Each trace is reduced to the
    set of (grid cell, absolute hour) pairs it visits. For each trial we pick a user, draw `p_points`
    random pairs from their trace and count how many users' traces contain all of them; the user is
    unique if the answer is 1. Coarser cells and sparser data make people harder to single out.
    """
    rng = np.random.default_rng(seed)
    sets = []
    for tr in traces:
        if len(tr.t) == 0:
            sets.append(set())
            continue
        cells = np.floor(tr.xy / cell_m).astype(int)
        hours = (tr.t // 3600).astype(int)
        sets.append(set(zip(cells[:, 0].tolist(), cells[:, 1].tolist(), hours.tolist())))
    users = [i for i, s in enumerate(sets) if len(s) >= p_points]
    if not users:
        return float("nan")
    unique = 0
    for _ in range(n_trials):
        u = users[rng.integers(len(users))]
        pool = list(sets[u])
        pick = [pool[i] for i in rng.choice(len(pool), p_points, replace=False)]
        unique += sum(all(q in s for q in pick) for s in sets) == 1
    return unique / n_trials
