"""Location-privacy defenses. Each takes a Trace and returns a new Trace."""
from __future__ import annotations

import numpy as np
from scipy.special import lambertw

from .synth import Trace


def planar_laplace_radius(epsilon, u):
    """Inverse CDF of the radial part of the planar Laplace distribution (Andres et al., 2013).
    epsilon is in 1/meter; returns radius in meters for uniform u in [0, 1)."""
    return -(np.real(lambertw((u - 1) / np.e, k=-1)) + 1) / epsilon


def geo_indistinguishability(trace, epsilon, rng):
    """Add planar Laplace noise independently to every point (epsilon-geo-indistinguishability)."""
    n = len(trace.t)
    r = planar_laplace_radius(epsilon, rng.random(n))
    th = rng.uniform(0, 2 * np.pi, n)
    noise = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return Trace(trace.user, trace.t, trace.xy + noise)


def sticky_geo_indistinguishability(trace, epsilon, cell_m, seed=0):
    """Geo-indistinguishability with *spatially correlated* noise.

    Independent per-point noise (see `geo_indistinguishability`) averages out when a user returns
    to the same place many times: the cluster centroid converges on the true location. Here the
    noise vector is a deterministic function of the grid cell (and a secret seed), so every visit
    to the same cell is displaced identically and averaging gains the attacker nothing. Points in
    different cells still get independent planar Laplace noise.
    """
    cells = np.floor(trace.xy / cell_m).astype(np.int64)
    uniq, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = inv.ravel()
    noise_per_cell = np.empty((len(uniq), 2))
    for i, (cx, cy) in enumerate(uniq):
        g = np.random.default_rng([seed, int(cx) + (1 << 31), int(cy) + (1 << 31)])
        r = planar_laplace_radius(epsilon, g.random())
        th = g.uniform(0, 2 * np.pi)
        noise_per_cell[i] = (r * np.cos(th), r * np.sin(th))
    return Trace(trace.user, trace.t, trace.xy + noise_per_cell[inv])


def grid_snap(trace, cell_m):
    """Spatial cloaking: snap every point to the center of a cell_m x cell_m grid cell."""
    snapped = (np.floor(trace.xy / cell_m) + 0.5) * cell_m
    return Trace(trace.user, trace.t, snapped)


def downsample(trace, every_s):
    """Temporal thinning: keep one point per `every_s` seconds."""
    bucket = np.floor(trace.t / every_s).astype(int)
    _, first = np.unique(bucket, return_index=True)
    return Trace(trace.user, trace.t[first], trace.xy[first])


def suppress_sensitive(trace, home_radius_m):
    """Drop every point within home_radius_m of the user's true home (a common 'privacy zone')."""
    keep = np.linalg.norm(trace.xy - trace.user.home, axis=1) > home_radius_m
    return Trace(trace.user, trace.t[keep], trace.xy[keep])


def suppress_random_radius(trace, r_min, r_max, seed=0):
    """Privacy zone with a random radius redrawn for every departure and every arrival.

    A fixed-radius zone leaks its own center: the first visible point is always roughly
    `radius` away from home, so the hole attack can walk back by that amount. Drawing a fresh
    radius in [r_min, r_max] for each half-day makes that distance unpredictable.
    """
    rng = np.random.default_rng(seed)
    half = (trace.t // 43200).astype(int)
    radii = {h: rng.uniform(r_min, r_max) for h in np.unique(half)}
    r = np.array([radii[h] for h in half])
    keep = np.linalg.norm(trace.xy - trace.user.home, axis=1) > r
    return Trace(trace.user, trace.t[keep], trace.xy[keep])


def suppress_and_noise(trace, radius_m, epsilon, rng):
    """Privacy zone followed by planar Laplace noise on the points that remain.

    The zone hides the stay at home; the noise destroys the collinearity of the departure and
    arrival routes that the hole attack uses to triangulate the hole's center.
    """
    return geo_indistinguishability(suppress_sensitive(trace, radius_m), epsilon, rng)
