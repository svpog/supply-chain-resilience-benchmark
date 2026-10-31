# Reproducing the results of the article

Every number, table and figure of the article is produced by the commands below. Two stages
require simulation; the rest is post-processing of the stored runs. Running all simulations
takes a few hours on a laptop; the stored results in `results/raw/` make the post-processing
steps immediate.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # unit tests of generators, flow computation and metrics
```

## 1. Simulations

```bash
python main.py --series S1  --runs 100     # Stage 1: topology
python main.py --series S2  --runs 100     # Stage 2: redundancy
python main.py --series S4  --runs 100     # Stage 3: disruption type and intensity
python main.py --series S5  --runs 100     # Stage 4: network size
python main.py --series S6  --runs 100     # Stage 5.A-5.C and 5.D
```

Runs added during the revision:

```bash
python main.py --series S1      --runs 1000   # convergence of the tail estimate
python main.py --series S5mu30  --runs 100    # repair rate proportional to N
python main.py --series S5mu50  --runs 100
python main.py --series S5mu80  --runs 100
python main.py --series S5mu120 --runs 100
```

Output of each series: `results/raw/<SERIES>/{raw_results.csv, Q_curves.parquet,
run_metadata.json}` and `results/aggregated/<SERIES>/aggregated_results.csv`.

## 2. Analysis

```bash
python _s1_calibration.py          # Stage 1 descriptive statistics and figures
python _s2_diagnostics.py          # Stage 2, including the K_res curve
python _s2_robustness.py           # Stage 2: effect of redundancy on robustness
python _s4_analysis.py             # Stage 3: heat map and hypothesis tests
python _s4_equivalence.py          # Stage 3: equivalence testing, cluster vs node
python _s5_analysis.py             # Stage 4: ranking stability under scaling
python _s6d_irreversibility.py     # Stage 5.D: irreversible damage
python _statistical_appendix.py    # full statistical appendix for all stages
python _topology_resources.py      # resource table of the four topologies
python _closeout_driver.py         # revision runs: inclusive sample, variance, repair rate
python _closeout_report.py         # consolidated report of the revision runs
```

## 3. Where each element of the article comes from

| Element of the article | Produced by | File |
|---|---|---|
| Table 1 (comparison with primary studies) | compiled manually from the cited sources | — |
| Table 2 (experimental stages) | — | — |
| Table 3 (Stage 1 descriptive statistics) | `_closeout_driver.py`, task R1-3 | `results/aggregated/closeout_r1_3_inclusive.csv` |
| Table 4 (irreversible damage share) | `_s6d_irreversibility.py` | `results/aggregated/S6D_irreversibility_report.txt` |
| Fig. 1 (mean Q(t) with IQR) | `_s1_calibration.py` | `results/figures/Q_curves_S1_canonical.png` |
| Fig. 2 (lower-tail resilience by topology) | `_s1_calibration.py` | `results/figures/cvar_comparison_S1_canonical.png` |
| Fig. 3 (redundancy level) | `_s2_diagnostics.py` | `results/figures/S2_K_res_curve.png` |
| Fig. 4 (heat map) | `_s4_analysis.py` | `results/figures/S4_heatmap_v2_en.png` |
| Fig. 5 (ranking stability) | `_s5_analysis.py` | `results/figures/S5_ranking_stability.png` |
| Fig. 6 (metric sensitivity) | `s6_*` sections of the S6 run | `results/figures/S6_{alpha,T,omega}_sensitivity.png` |
| Fig. 7 (effect of w on resilience) | `_s6d_irreversibility.py` | `results/figures/S6D_w_sensitivity.png` |
| Fig. 8 (effect of w on recovery) | `_s6d_irreversibility.py` | `results/figures/S6D_recovery_extent.png` |
| Kruskal-Wallis, Mann-Whitney, Cliff's delta, bootstrap intervals | `_statistical_appendix.py` | `results/aggregated/S_statistical_appendix.txt` |
| Equivalence of cluster and node failure | `_s4_equivalence.py` | `results/aggregated/S4_cluster_vs_node_equivalence.txt` |
| Resource table of the topologies | `_topology_resources.py` | `results/aggregated/topology_resources.txt` |
| Convergence of the tail estimate, inclusive sample, variance decomposition, repair rate | `_closeout_driver.py` and `_closeout_report.py` | `results/aggregated/closeout_report.txt` |

## 4. Statistical conventions

* Global comparisons: Kruskal-Wallis. Pairwise: Mann-Whitney with Bonferroni correction over
  the stated family of hypotheses; adjusted p-values are capped at 1.
* Effect size: Cliff's delta for independent samples, matched-pairs rank-biserial correlation
  for the paired design of Stage 5.D. Thresholds after Romano (2006): 0.147 / 0.33 / 0.474.
* Confidence intervals: percentile bootstrap, 10 000 resamples, seed 20261031.
* Lower-tail mean: the tail consists of exactly `max(1, floor(alpha * n))` smallest values after
  sorting, so that observations equal to the boundary value beyond that count stay outside the
  tail. The estimate does not depend on the order of records.
* Spearman correlations between rankings are computed on four points; with `n = 4` the smallest
  attainable two-sided p-value is 0.083, so these correlations are reported without a p-value.

## 5. Notes on specific runs

* `results/raw/S4_old_r_clust/` holds the superseded Stage 3 run in which cluster intensity was
  parameterized by a fixed breadth-first radius. It is retained because the article refers to
  that parameterization as the reason why clustered damage appeared more severe: a radius of
  two covered about 80 % of the nodes of a dense network against the 20 % removed at
  `p_v = 0.20`. The published results use the target-share parameterization.
* The control run of the repair rate (`S5mu*`) uses its own seed grid, so its absolute values
  differ from Stage 4 in the third decimal place. Both repair regimes share that grid, which
  makes the comparison between them paired.
