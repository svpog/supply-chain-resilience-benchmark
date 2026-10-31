# Supply chain resilience under stochastic disruptions

Code, configurations and experimental results supporting the article

> S. V. Pohorielov, O. V. Yanovskyi. **Supply chain resilience under stochastic disruptions**.
> *Applied Aspects of Information Technology*, 2026, Vol. 9, No. 4.
> DOI: `10.15276/aait.09.2026.XX` *(to be completed on publication)*

A supply chain is represented as a directed capacitated graph. Functionality is the
normalized maximum flow from a supersource to a supersink; resilience is evaluated over the
whole performance trajectory by an integral index, by component metrics of robustness and
recovery, and by a lower-tail (CVaR-like) risk metric under a Monte Carlo protocol.

Everything reported in the article can be regenerated from this repository. See
[`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the command behind every table and figure.

## Quick start

Python 3.10 or newer (the code was run on 3.10 and 3.13).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py --series S1 --runs 100      # Stage 1 of the article: topology
pytest -q                                   # unit tests of generators, flow and metrics
```

Each run writes `results/raw/<SERIES>/raw_results.csv`, the trajectories
`Q_curves.parquet`, a `run_metadata.json` with the exact parameters and seeds, and the
aggregated table `results/aggregated/<SERIES>/aggregated_results.csv`.

## Series numbering

The article groups the experiments into five stages. The code and the result files use the
numbering of the internal experimental protocols, which is not identical: protocol series S3
(hub centralization) belongs to a companion study and is not part of this article. The
correspondence is:

| Article stage | Code series | Varied factor | Config file |
|---|---|---|---|
| Stage 1 | `S1` | topology type | `config/series_S1.yaml` |
| Stage 2 | `S2` | redundancy level `K_res` | `config/series_S2.yaml` |
| Stage 3 | `S4` | disruption type and intensity | `config/series_S4.yaml` |
| Stage 4 | `S5` | network size `N` | `config/series_S5.yaml` |
| Stage 5.A–5.C | `S6` | metric parameters α, T_eff, ω | `config/series_S6.yaml` |
| Stage 5.D | `S6D` | irreversible damage share `w` | section `S6D` of `series_S6.yaml` |

Two further sets of runs were produced during the revision and are included here:
`S1_M1000` (Stage 1 repeated with 1000 runs, for the convergence of the tail estimate) and
`S5mu30 … S5mu120` (Stage 4 with the repair rate scaled proportionally to network size).

## Layout

```
src/                  model, disruptions, recovery, metrics, analysis, visualization
  network/            generators of the four topologies, flow model, supersource/supersink
  disruptions/        node failure, edge degradation, cluster damage
  recovery/           baseline random recovery and the strategy interface
  metrics/            integral, structural and stochastic metrics
  experiments/        the runner: one Monte Carlo replication end to end
config/               base configuration and one YAML per series
_*.py                 post-processing scripts: hypothesis tests, appendices, diagnostics
results/raw/          per-run records and trajectories for every series of the article
results/aggregated/   descriptive statistics, hypothesis tests, statistical appendix
results/figures/      the figures of the article
docs/                 mathematical specification, its patches, reproduction guide
tests/                unit tests
```

## Model in brief

* **Network.** Directed graph `G = (V, E, c)` with supplier set `V_s` and consumer set `V_c`.
  A supersource is joined to each supplier by an arc whose capacity is that supplier's supply
  volume, and each consumer to a supersink by an arc whose capacity is its demand. Capacities
  are assigned by a capacity-aware rule (`src/network/base.py`) so that the undisturbed
  network is not limited by an arbitrary split of supply.
* **Functionality.** `Q(t) = F(t) / F₀`, where `F₀` is the maximum flow of the **undisturbed**
  network. The state `t = 0` is undisturbed, so `Q(0) = 1`; the disruption is applied between
  `t = 0` and `t = 1`.
* **Disruptions.** Node failure: each node fails independently with probability `p_v`. Edge
  degradation: each arc degrades independently with probability `p_e`, its capacity multiplied
  by `(1 − γ)`. Cluster: breadth-first growth from a uniformly chosen epicentre until a target
  share of nodes is covered, the last level truncated by a uniform random subset.
* **Recovery.** One element per step, drawn uniformly among the repairable damaged elements,
  with failed nodes taking strict precedence over degraded arcs. A repair is complete;
  repairing a node does **not** restore the capacity of its incident arcs. Irreversibly
  damaged elements are never restored.
* **Metrics.** `ρ` = mean of the trajectory over all `T + 1` steps; `R_rb` = minimum `Q`;
  `R_rp` = recovery rate between the event time and the recovery time; `R_inf = Q(T)`.
  Across runs: median `ρ_med`, interquartile range `ρ_IQR`, and the lower-tail mean
  `ρ_CVaR,α` over the `max(1, ⌊α·n⌋)` smallest values.

## Reproducibility

Each run derives three independent sub-seeds from one master seed: one for the network
realization, one for the disruption and one for the recovery process. The master seed grid is
fixed per series in the configuration, so a series reproduces exactly on re-execution. A run
whose baseline flow is zero cannot be normalized and is flagged as invalid; a run in which the
disruption destroyed all active sources or consumers while the baseline flow was positive is
simulated and retained. Both flags are recorded per run in `raw_results.csv`.

## Licence

Code: MIT. Data under `results/`: CC BY 4.0. See [`LICENSE`](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the article; if you need to cite the software
itself, use the archived release and its DOI.
