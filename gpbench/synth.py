"""Synthetic but structured mobility: users with a home, a workplace and a few leisure spots.

Each simulated day: sleep at home, commute to work on weekdays, optionally visit a leisure
place, come home. GPS is sampled every `step_s` seconds with Gaussian jitter. Ground truth
home/work are kept so attacks can be scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class User:
    uid: int
    home: np.ndarray
    work: np.ndarray
    leisure: list = field(default_factory=list)


@dataclass
class Trace:
    user: User
    t: np.ndarray    # seconds since start
    xy: np.ndarray   # (n, 2) meters


def make_users(n, rng, city_radius=8000.0):
    users = []
    for uid in range(n):
        home = rng.uniform(-city_radius, city_radius, 2)
        work = rng.uniform(-city_radius, city_radius, 2)
        while np.linalg.norm(work - home) < 1500:      # avoid trivially co-located home/work
            work = rng.uniform(-city_radius, city_radius, 2)
        leisure = [rng.uniform(-city_radius, city_radius, 2) for _ in range(3)]
        users.append(User(uid, home, work, leisure))
    return users


def _grid_start(t0, step):
    """First tick of the global sampling clock at or after t0 (GPS ticks are not synced to departures)."""
    return np.ceil(t0 / step) * step


def _stay(pos, t0, t1, step, rng, jitter):
    t = np.arange(_grid_start(t0, step), t1, step, dtype=float)
    return t, pos + rng.normal(0, jitter, (len(t), 2))


def _move(a, b, t0, speed, step, rng, jitter):
    d = float(np.linalg.norm(b - a))
    dur = max(d / speed, step)
    t = np.arange(_grid_start(t0, step), t0 + dur, step, dtype=float)
    frac = ((t - t0) / dur)[:, None]
    return t, a + (b - a) * frac + rng.normal(0, jitter, (len(t), 2)), t0 + dur


def simulate(user, days, rng, step_s=300, jitter=8.0, speed=8.0):
    """Return a Trace covering `days` days (day 0 is a Monday)."""
    ts, ps = [], []
    day_s = 86400
    for d in range(days):
        base = d * day_s
        weekday = d % 7 < 5
        cur_t = base
        if weekday:
            leave = base + 8 * 3600 + rng.normal(0, 900)
            t, p = _stay(user.home, cur_t, leave, step_s, rng, jitter); ts.append(t); ps.append(p)
            t, p, cur_t = _move(user.home, user.work, leave, speed, step_s, rng, jitter); ts.append(t); ps.append(p)
            end = base + 17 * 3600 + rng.normal(0, 1200)
            t, p = _stay(user.work, cur_t, end, step_s, rng, jitter); ts.append(t); ps.append(p)
            cur_t, here = end, user.work
        else:
            here = user.home
            leave = base + 11 * 3600 + rng.normal(0, 1800)
            t, p = _stay(user.home, cur_t, leave, step_s, rng, jitter); ts.append(t); ps.append(p)
            cur_t = leave
        if rng.random() < 0.5:                             # leisure visit
            spot = user.leisure[rng.integers(len(user.leisure))]
            t, p, cur_t = _move(here, spot, cur_t, speed, step_s, rng, jitter); ts.append(t); ps.append(p)
            stay_end = cur_t + rng.uniform(1.5, 3) * 3600
            t, p = _stay(spot, cur_t, stay_end, step_s, rng, jitter); ts.append(t); ps.append(p)
            cur_t, here = stay_end, spot
        t, p, cur_t = _move(here, user.home, cur_t, speed, step_s, rng, jitter); ts.append(t); ps.append(p)
        t, p = _stay(user.home, cur_t, base + day_s, step_s, rng, jitter); ts.append(t); ps.append(p)
    t = np.concatenate(ts); xy = np.vstack(ps)
    order = np.argsort(t, kind="stable")
    return Trace(user, t[order], xy[order])
