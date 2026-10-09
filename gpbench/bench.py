"""Run the benchmark: every defense configuration against the same simulated population."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import defenses as D
from .attacks import hole_attack
from .attacks import stay_points
from .metrics import privacy_scores, uniqueness, utility_scores
from .synth import make_users, simulate


@dataclass
class Config:
    name: str
    family: str
    param: float
    apply: Callable            # (trace, rng) -> trace
    hole_radius: float = 0.0   # > 0: the attacker also runs the hole attack assuming this zone radius


def default_configs():
    cfgs = [Config("none", "none", 0.0, lambda t, r: t)]
    for eps in (0.1, 0.02, 0.01, 0.005):  # 1/m: expected radius = 2/eps -> 20 m ... 400 m
        cfgs.append(Config(f"geo-ind eps={eps}/m", "geo-ind", eps, lambda t, r, e=eps: D.geo_indistinguishability(t, e, r)))
    for eps in (0.02, 0.01, 0.005):
        cfgs.append(Config(f"sticky geo-ind eps={eps}/m", "sticky", eps,
                           lambda t, r, e=eps: D.sticky_geo_indistinguishability(t, e, 100.0, seed=int(r.integers(1 << 31)))))
    for cell in (200, 500, 1000, 2000):
        cfgs.append(Config(f"grid {cell} m", "grid", cell, lambda t, r, c=cell: D.grid_snap(t, c)))
    for every in (900, 3600):
        cfgs.append(Config(f"downsample {every // 60} min", "downsample", every, lambda t, r, e=every: D.downsample(t, e)))
    for rad in (300, 600):
        cfgs.append(Config(f"home zone {rad} m", "zone", rad, lambda t, r, x=rad: D.suppress_sensitive(t, x), hole_radius=rad))
    cfgs.append(Config("home zone random 300-1500 m", "zone-rand", 900,
                       lambda t, r: D.suppress_random_radius(t, 300, 1500, seed=int(r.integers(1 << 31))),
                       hole_radius=900))
    cfgs.append(Config("home zone 600 m + noise eps=0.01/m", "zone-noise", 600,
                       lambda t, r: D.suppress_and_noise(t, 600, 0.01, r), hole_radius=600))
    return cfgs


_STATE = {}


def _evaluate(idx):
    cfg, traces, seed = _STATE["cfgs"][idx], _STATE["traces"], _STATE["seed"]
    rng = np.random.default_rng([seed, idx])           # per-config stream: results do not depend on worker order
    rows_p, rows_u, prots = [], [], []
    for tr, sp in zip(traces, _STATE["orig_sp"]):
        prot = cfg.apply(tr, np.random.default_rng(rng.integers(1 << 31)))
        prots.append(prot)
        rows_p.append(privacy_scores(prot, tr.user.home, tr.user.work))
        rows_u.append(utility_scores(tr, prot, orig_sp=sp))
    he = np.array([r["home_err"] for r in rows_p])
    naive = float(np.mean([r["home_hit"] for r in rows_p]))
    adaptive = naive
    if cfg.hole_radius:
        errs = []
        for tr, prot in zip(traces, prots):
            h = hole_attack(prot, cfg.hole_radius)
            errs.append(float(np.linalg.norm(h - tr.user.home)) if h is not None else np.inf)
        adaptive = float(np.mean(np.array(errs) <= 200.0))
    return {
        "config": cfg.name, "family": cfg.family, "param": cfg.param,
        "home_hit_rate": naive,
        "home_hit_rate_adaptive": adaptive,
        "work_hit_rate": float(np.mean([r["work_hit"] for r in rows_p])),
        "home_median_err_m": float(np.median(he)) if np.isfinite(he).any() else None,
        "stay_recall": float(np.nanmean([r["stay_recall"] for r in rows_u])),
        "length_ratio": float(np.nanmean([r["length_ratio"] for r in rows_u])),
        "mean_displacement_m": float(np.nanmean([r["mean_displacement"] for r in rows_u])),
        "unique_4pts_1km": uniqueness(prots, 4, 1000.0, seed=seed),
    }


def run(n_users=30, days=14, seed=0, configs=None, workers=None):
    """Simulate a population once, then evaluate every defense config (in parallel processes)."""
    import multiprocessing as mp
    import os

    rng = np.random.default_rng(seed)
    users = make_users(n_users, rng)
    traces = [simulate(u, days, rng) for u in users]
    cfgs = configs or default_configs()
    _STATE.update(cfgs=cfgs, traces=traces, seed=seed, orig_sp=[stay_points(t) for t in traces])
    workers = workers or min(os.cpu_count() or 1, len(cfgs))
    if workers > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_evaluate, range(len(cfgs)))
    else:
        results = [_evaluate(i) for i in range(len(cfgs))]
    return {"n_users": n_users, "days": days, "seed": seed, "results": results}


def to_markdown(report):
    lines = [f"Population: {report['n_users']} users x {report['days']} days, seed {report['seed']}", "",
             "| defense | home found (stay-point attack) | home found (adaptive attack) | work found | "
             "stay recall | users unique from 4 points |",
             "|---|---|---|---|---|---|"]
    for r in report["results"]:
        lines.append(f"| {r['config']} | {r['home_hit_rate']:.0%} | {r['home_hit_rate_adaptive']:.0%} | "
                     f"{r['work_hit_rate']:.0%} | {r['stay_recall']:.0%} | {r['unique_4pts_1km']:.0%} |")
    return "\n".join(lines)


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"none": "k", "geo-ind": "tab:blue", "sticky": "tab:purple", "grid": "tab:green",
              "downsample": "tab:orange", "zone": "tab:red", "zone-rand": "tab:brown", "zone-noise": "tab:pink"}
    seen, placed = set(), []
    for i, r in enumerate(report["results"]):
        x, y = r["stay_recall"], 1 - r["home_hit_rate_adaptive"]
        ax.scatter(x, y, c=colors[r["family"]], s=70, label=r["family"] if r["family"] not in seen else None, zorder=3)
        seen.add(r["family"])
        if any(abs(x - px) < 0.07 and abs(y - py) < 0.05 for px, py in placed):
            continue                                  # crowded corner: skip the label, keep the point
        placed.append((x, y))
        ax.annotate(r["config"].replace("geo-ind ", "").replace("home zone ", "zone "), (x, y), fontsize=7,
                    xytext=(5, 5 if i % 2 else -9), textcoords="offset points")
    ax.set_xlabel("utility: stay-point recall (higher is better)")
    ax.set_ylabel("privacy: share of homes NOT recovered by the best attack (higher is better)")
    ax.set_title("Privacy-utility trade-off of GPS defenses")
    ax.set_xlim(-0.03, 1.12); ax.set_ylim(-0.05, 1.08)
    ax.grid(alpha=.3); ax.legend(loc="center left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def uniqueness_curve(n_users=30, days=14, seed=0, cells=(500, 1000, 2000, 5000, 10000), points=(1, 2, 3, 4)):
    """Share of users singled out by p known (place, hour) points, for several grid cell sizes (raw traces)."""
    rng = np.random.default_rng(seed)
    traces = [simulate(u, days, rng) for u in make_users(n_users, rng)]
    rows = ["| cell size | " + " | ".join(f"{p} point{'s' if p > 1 else ''}" for p in points) + " |",
            "|---|" + "---|" * len(points)]
    for c in cells:
        vals = " | ".join(f"{uniqueness(traces, p, float(c), n_trials=300, seed=seed + 1):.0%}" for p in points)
        rows.append(f"| {c / 1000:g} km | {vals} |")
    return "\n".join(rows)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="gpbench", description="GPS privacy defense benchmark")
    p.add_argument("--users", type=int, default=30)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="write full results to this file")
    p.add_argument("--plot", help="write privacy-utility scatter plot (png or svg)")
    p.add_argument("--uniqueness", action="store_true", help="print only the re-identification curve for raw traces")
    a = p.parse_args(argv)
    if a.uniqueness:
        print(uniqueness_curve(a.users, a.days, a.seed))
        return
    report = run(a.users, a.days, a.seed)
    print(to_markdown(report))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=2)
    if a.plot:
        plot(report, a.plot)


if __name__ == "__main__":
    main()
