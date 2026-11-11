# Time-varying recalibration of density forecast pools

Code, data and figures for recalibrating combined density forecasts over time. A forecast pool
(the Loss Discounting Framework pool, or Dynamic Bayesian Predictive Synthesis) produces a
predictive distribution each period. We map its probability integral transforms (PITs) through a
time-varying recalibration function so the corrected forecasts are better calibrated and score
better out of sample.

The repository covers:

- **Simulations**: replications across several data-generating scenarios and seeds.
- **Exchange rates**: nine currencies against the USD (AUD, CAD, CHF, EUR, GBP, JPY, NOK, NZD, SEK).
- **US inflation**: a pool of four forecasting models.

## Methods

| Component | Where | What it does |
|---|---|---|
| LDF pool | `ldf.py` | Loss Discounting Framework: combines the agents' densities with discounted log-score weights over a grid of discount factors. |
| Dynamic BPS | `bps.py` | Python port of McAlinn & West's MATLAB code for Dynamic Bayesian Predictive Synthesis (*J. Econometrics*, 2019). |
| PITs | `pit.py` | PITs from discretised CDFs, plus calibration diagnostics. |
| Spline recalibration | `recal_dma_new.py` | Recalibration with a rational-quadratic spline. Two variants: a score-driven (GAS) update, and a convolution / neural-network update. Both are combined across discount factors by dynamic model averaging (DMA). |
| Beta recalibration | `recal_beta_dma_new.py` | The same two variants with a Beta link in place of the spline. Shares preprocessing and scoring with the spline module. |
| Evaluation | `localised_metrics.py`, `metrics.py`, `uniformity_tests.py`, `Log_likelihood.py` | Log score, CRPS, interval scores, ECE, uniformity tests (KS, AD, Ljung-Box, etc.), localised scores by region, and Diebold-Mariano / Giacomini-White tests with HAC errors. |
| Reporting | `sim_report.py`, `replicate_sim.py`, `switch_event_study.py`, `beta_parameter_plot.py` | Simulation design, replications, LaTeX tables and paper figures. |
| Spline backend | `rational_quadratic.py`, `nn.py`, `utils/` | Rational-quadratic spline and residual network, adapted from [nflows](https://github.com/bayesiains/nflows). |

## Repository layout

```
.
├── paper_simulations_v3.ipynb   # simulations: replications, tables, figures
├── fx_all_currencies.ipynb      # FX: recalibration vs benchmarks, all currencies
├── fx_bps.ipynb                 # FX (GBP): recalibrating Dynamic BPS vs the LDF pool
├── fx_bps_all.ipynb             # FX: BPS vs LDF, all currencies
├── inflation_recal.ipynb        # US inflation: recalibration vs benchmarks
├── inflation_bps_new.ipynb      # US inflation: recalibrating Dynamic BPS vs the LDF pool
├── *.py                         # method, evaluation and reporting modules (see above)
├── utils/                       # spline helper functions
├── data/                        # input data (see below)
├── figures/
│   ├── fx/
│   ├── inflation/
│   └── simulation/
└── results/                     # cached runs (.pkl), created by the notebooks, not tracked
```

## Data

| File | Contents |
|---|---|
| `data/Y.xlsx` | Exchange-rate series for the nine currencies. |
| `data/uip_l1_save_{loglik,means,covariance}.npz` | Model forecasts for the exchange-rate pool: log-likelihoods, predictive means and covariances. |
| `data/data.xlsx` | US inflation, with each of the four models' predictive mean, variance and degrees of freedom (sheets `Inflation`, `Agent Mean`, `Agent Var`, `Agent dof`). |

## Getting started

```bash
git clone https://github.com/callum-0214/recalibration.git
cd recalibration
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks from the repository root, so the module imports and relative paths
(`data/`, `figures/`, `results/`) resolve.

**Caching and run time.** Expensive steps are cached under `results/`: Dynamic BPS MCMC runs,
per-currency recalibrations and the simulation replications. Delete a cache file to recompute
it. On first run:

- `paper_simulations_v3.ipynb`, section 1, takes about 2 hours for 10 seeds and writes
  `results/paper_reps_v5.pkl`. Later sections read only from that file.
- The BPS notebooks run 3,000 burn-in and 5,000 MCMC iterations per forecast origin, and cache
  each run as `results/bps_*_3000_8000.pkl`.

## Acknowledgements

- `bps.py` ports the MATLAB code accompanying McAlinn, K. and West, M. (2019), "Dynamic Bayesian
  predictive synthesis in time series forecasting", *Journal of Econometrics* 210, 155–169.
- `rational_quadratic.py`, `nn.py` and `utils/` are adapted from
  [nflows](https://github.com/bayesiains/nflows) (Durkan, Bekasov, Murray and Papamakarios, MIT
  licence).
