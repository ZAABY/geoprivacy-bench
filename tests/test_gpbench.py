import numpy as np
import pytest

from gpbench import defenses as D
from gpbench.attacks import hole_attack, infer_home_work
from gpbench.geo import dist, to_latlon, to_xy
from gpbench.metrics import privacy_scores, uniqueness, utility_scores
from gpbench.synth import make_users, simulate


@pytest.fixture(scope="module")
def trace():
    rng = np.random.default_rng(1)
    return simulate(make_users(1, rng)[0], 10, rng)


@pytest.fixture(scope="module")
def population():
    rng = np.random.default_rng(7)
    return [simulate(u, 10, rng) for u in make_users(40, rng)]


def test_projection_roundtrip():
    xy = to_xy([52.52, 52.53], [13.40, 13.42], 52.5, 13.4)
    lat, lon = to_latlon(xy, 52.5, 13.4)
    assert np.allclose(lat, [52.52, 52.53], atol=1e-9) and np.allclose(lon, [13.40, 13.42], atol=1e-9)
    assert dist(xy[0], xy[1]) == pytest.approx(1752, rel=0.01)  # 0.01 deg lat = 1112 m, 0.02 deg lon at 52.5N = 1355 m


def test_synthetic_trace_is_sorted_and_attack_finds_truth(trace):
    assert np.all(np.diff(trace.t) >= 0)
    home, work = infer_home_work(trace)
    assert dist(home, trace.user.home) < 50 and dist(work, trace.user.work) < 50


def test_planar_laplace_mean_radius():
    eps = 0.01
    r = D.planar_laplace_radius(eps, np.random.default_rng(0).random(200000))
    assert r.mean() == pytest.approx(2 / eps, rel=0.02)  # E[r] = 2/eps for the planar Laplace


def test_independent_noise_is_averaged_out_but_sticky_noise_is_not(trace):
    eps = 0.01  # expected displacement 200 m
    h_i, _ = infer_home_work(D.geo_indistinguishability(trace, eps, np.random.default_rng(0)))
    h_s, _ = infer_home_work(D.sticky_geo_indistinguishability(trace, eps, 100.0, seed=0))
    err_i = dist(h_i, trace.user.home) if h_i is not None else np.inf
    err_s = dist(h_s, trace.user.home) if h_s is not None else np.inf
    assert err_i < 40
    assert err_s > err_i


def test_sticky_noise_is_deterministic_per_cell(trace):
    a = D.sticky_geo_indistinguishability(trace, 0.01, 100.0, seed=5)
    b = D.sticky_geo_indistinguishability(trace, 0.01, 100.0, seed=5)
    assert np.array_equal(a.xy, b.xy)


def test_grid_snap_and_downsample(trace):
    g = D.grid_snap(trace, 500)
    assert np.all(np.abs(g.xy - trace.xy) <= 500 * 0.5 * np.sqrt(2) + 1e-6)
    d = D.downsample(trace, 3600)
    assert len(d.t) < len(trace.t) and len(np.unique(np.floor(d.t / 3600))) == len(d.t)


def test_home_zone_fools_stay_point_attack_but_not_hole_attack(trace):
    z = D.suppress_sensitive(trace, 400)
    s = privacy_scores(z, trace.user.home, trace.user.work)
    assert not s["home_hit"] and s["work_hit"]
    est = hole_attack(z, 400)
    assert dist(est, trace.user.home) < 200


def test_hole_attack_survives_random_radius_but_not_added_noise(population):
    """Randomizing the zone radius is not enough: routes in different directions still triangulate home."""
    rand = [dist(hole_attack(D.suppress_random_radius(t, 300, 1500, seed=i), 900), t.user.home)
            for i, t in enumerate(population)]
    noisy = [dist(hole_attack(D.suppress_and_noise(t, 600, 0.01, np.random.default_rng(i)), 600), t.user.home)
             for i, t in enumerate(population)]
    assert np.median(rand) < 200
    assert np.median(noisy) > 3 * np.median(rand)


def test_utility_identity(trace):
    u = utility_scores(trace, trace)
    assert u["length_ratio"] == pytest.approx(1) and u["stay_recall"] == 1 and u["mean_displacement"] == 0


def test_uniqueness_drops_with_coarser_cells_and_grows_with_points(population):
    fine_4 = uniqueness(population, 4, 250.0, n_trials=120)
    assert fine_4 >= uniqueness(population, 1, 250.0, n_trials=120)
    assert fine_4 >= uniqueness(population, 4, 20000.0, n_trials=120)
    assert uniqueness(population, 4, 20000.0, n_trials=120) < fine_4 - 0.3
