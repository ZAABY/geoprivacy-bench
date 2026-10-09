# geoprivacy-bench

A benchmark and attack toolkit for **GPS location privacy**. It simulates people with a home, a workplace and a few leisure spots, applies privacy defenses to their GPS traces, then runs attackers that try to recover where they live and work. Every defense gets a privacy score (can the attacker still find home?) and a utility score (does the trace still show where the person stays?).

Your phone's location history is one of the most identifying datasets there is. Common defenses (add noise, round to a grid, hide a zone around home) are often presented as safe. This project measures how safe they actually are against an attacker who is allowed to be clever.

## Findings

Run with 30 simulated users, 14 days each, GPS sample every 5 minutes (seed 42). "Found" means the attacker's estimate lands within 200 m of the true location. Full output: [docs/results.md](docs/results.md), raw numbers: [docs/results.json](docs/results.json).

![privacy utility trade-off](docs/privacy_utility.png)

Population: 30 users x 14 days, seed 42

| defense | home found (stay-point attack) | home found (adaptive attack) | work found | stay recall | users unique from 4 points |
|---|---|---|---|---|---|
| none | 100% | 100% | 100% | 100% | 100% |
| geo-ind eps=0.1/m | 100% | 100% | 100% | 96% | 100% |
| geo-ind eps=0.02/m | 100% | 100% | 100% | 94% | 100% |
| geo-ind eps=0.01/m | 100% | 100% | 100% | 91% | 98% |
| geo-ind eps=0.005/m | 97% | 97% | 97% | 79% | 100% |
| sticky geo-ind eps=0.02/m | 90% | 90% | 90% | 93% | 100% |
| sticky geo-ind eps=0.01/m | 63% | 63% | 57% | 74% | 100% |
| sticky geo-ind eps=0.005/m | 23% | 23% | 23% | 51% | 100% |
| grid 200 m | 100% | 100% | 100% | 92% | 100% |
| grid 500 m | 63% | 63% | 43% | 79% | 100% |
| grid 1000 m | 17% | 17% | 7% | 23% | 100% |
| grid 2000 m | 0% | 0% | 3% | 5% | 97% |
| downsample 15 min | 100% | 100% | 100% | 92% | 98% |
| downsample 60 min | 100% | 100% | 100% | 63% | 100% |
| home zone 300 m | 0% | 87% | 100% | 80% | 100% |
| home zone 600 m | 0% | 87% | 100% | 80% | 100% |
| home zone random 300-1500 m | 0% | 83% | 100% | 79% | 100% |
| home zone 600 m + noise eps=0.01/m | 0% | 43% | 100% | 71% | 98% |

What stands out:

1. **Per-point noise barely helps.** Geo-indistinguishability (planar Laplace noise, 200 m expected displacement at eps=0.01/m) still lets the stay-point attack find 100% of homes. People return to the same places again and again, so independent noise averages out and the cluster centroid converges on the truth.
2. **Correlated ("sticky") noise helps much more.** `sticky geo-ind` derives the noise from the grid cell, so every visit to the same place is displaced identically and averaging gains nothing. At eps=0.005/m, 77% of homes stay hidden at 51% stay-point recall. Grid cloaking at 1 km hides 83% of homes but keeps only 23% of the stays, so sticky noise is the better trade at similar privacy.
3. **Hiding a zone around home looks perfect and is not.** Against a naive attacker, a 300 m or 600 m home zone gives 0% home found. The *hole attack* ([gpbench/attacks.py](gpbench/attacks.py)) never looks inside the zone. It takes the first visible point after each overnight gap and its neighbour on the route, which gives a line pointing at home. Several routes in different directions triangulate the hole's center with a weighted least squares fit. It finds 87% of homes.
4. **Randomizing the zone radius does not fix it** (83% found). The radius only affects the position along each line, not where the lines cross.
5. **Adding noise on top of the zone does** (43% found): noise breaks the collinearity of the visible route points that the triangulation needs.
6. **Re-identification is easy.** A handful of known (place, hour) points is enough to single a person out of the crowd, in the spirit of de Montjoye et al., "Unique in the Crowd" (2013). For raw traces:

| cell size | 1 point | 2 points | 3 points | 4 points |
|---|---|---|---|---|
| 0.5 km | 92% | 99% | 100% | 100% |
| 1 km | 83% | 97% | 100% | 100% |
| 2 km | 54% | 91% | 95% | 99% |
| 5 km | 15% | 61% | 70% | 85% |
| 10 km | 0% | 8% | 20% | 24% |


The attack error numbers depend heavily on the simulation, see Limitations.

## What is in here

| file | what it does |
|---|---|
| `gpbench/synth.py` | structured mobility simulator (home, work, leisure; commutes; GPS jitter; sampling clock independent of departures) |
| `gpbench/defenses.py` | geo-indistinguishability (Lambert W sampler for the planar Laplace radius), sticky noise, grid cloaking, downsampling, privacy zone, random-radius zone, zone plus noise |
| `gpbench/attacks.py` | stay-point attack (DBSCAN plus dwell-time scoring for home and work) and the hole attack (route triangulation) |
| `gpbench/metrics.py` | attack success, stay-point recall, path-length ratio, displacement, re-identification uniqueness |
| `gpbench/bench.py` | runs every defense on the same population, in parallel; markdown, JSON and plot output |
| `tests/` | pytest suite, including checks that the Laplace sampler has the right mean radius (2/eps), sticky noise is deterministic per cell, and the hole attack beats the zone |

## Run it

```bash
pip install -r requirements.txt
pytest -q
python -m gpbench.bench --users 30 --days 14 --seed 42 --plot privacy_utility.png --json results.json
python -m gpbench.bench --uniqueness --users 30 --seed 42
```

The full benchmark takes about a minute on two CPU cores.

Use it from Python:

```python
import numpy as np
from gpbench.synth import make_users, simulate
from gpbench import defenses as D
from gpbench.attacks import infer_home_work, hole_attack

rng = np.random.default_rng(0)
user = make_users(1, rng)[0]
trace = simulate(user, days=14, rng=rng)
protected = D.suppress_sensitive(trace, 600)            # hide everything within 600 m of home
print(infer_home_work(protected))                       # naive attack: home is not found
print(hole_attack(protected, 600), user.home)           # hole attack: lands on the real home
```

## Limitations

- The population is **synthetic**: 30 users, one commute and a few leisure places each, straight-line movement and Gaussian jitter. Real mobility is messier, which makes some attacks harder and others easier. Treat the numbers as a comparison between defenses, not as absolute risk.
- The sampling interval is 5 minutes at 8 m/s, so route points are 2.4 km apart. Denser sampling makes the hole attack more accurate.
- The hole attack assumes the attacker knows the zone radius and the sampling rate (Kerckhoffs' principle).
- Utility is measured as stay-point recall only. Other uses of location data (routing, traffic) would need other metrics.
- Geo-indistinguishability is analysed as independent per-point noise. Its formal guarantee holds per point, which is exactly why correlated observations break it.
- Not tested on real traces yet. A CSV loader for datasets like GeoLife is the obvious next step.

## License

MIT
