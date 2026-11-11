"""Replications for the paper: every scenario, every discount factor, R seeds.

WHAT IS REPLICATED AND WHAT IS NOT
  Replicated, averaged across seeds (inference):
    * test heatmap      mean number of the 12 discount factors passing each test
    * scores table      KS, LogS, CRPS, IS90/95, CSL_L/R at one delta: mean over seeds,
                        stars from the across-seed t-test (seeds are independent)
    * Con - GAS         per scenario (and by regime for TV dispersion)
    * TVD evidence      event study pooled over every seed's switches; mean cumulative
                        log-score gain with a band across seeds
  From ONE representative seed (illustration -- averaging blurs exactly what they show):
    * raw / recalibrated PIT histograms, ECDF-minus-uniform, PIT panel
    * Beta parameter paths (switch times differ across seeds, so paths cannot be averaged)

Notebook:
    import replicate_sim as rp
    reps = rp.run_all(seeds=range(10), n_jobs=4, path="paper_reps.pkl")
    reps = rp.load("paper_reps.pkl")
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest, t as student_t
from sim_report import COLORS

PHI, SIGMA_X, SIGMA_Y, SIGMA_Z, K, T = 0.9, 0.5, 0.5, 0.7, 20, 2001
DISCOUNT_FACTORS = np.array([1, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.001])
FIXED_LEVELS = np.asarray([0]*50 + [-1]*50 + [1]*50 + [-1]*50 + [0]*200 + [-1]*400 + [0]*50 + [-1]*50
                          + [1]*50 + [-1]*10 + [1]*10 + [0]*10 + [-1]*10 + [1]*10 + [0]*50 + [1]*50
                          + [0]*50 + [-1]*50 + [1]*400 + [0]*50 + [-1]*50 + [1]*50 + [0]*50 + [0]*201)

# notebook key -> get_forecasts scenario ("markov" = the separate Markov-switching generator)
SCENARIOS = {
    "Over_dispersed": "original",
    "Under_dispersed": "under_dispersed",
    "Biased": "biased",
    "Bimodal": "bimodal",
    "Trimodal": "trimodal",
    "Time_varying_dispersion": "time_varying_dispersion",
    "Time_varying_markov_switch": "markov",
}
DISPLAY = {"Over_dispersed": "Over dispersed", "Under_dispersed": "Under dispersed",
           "Biased": "Biased", "Bimodal": "Bimodal", "Trimodal": "Trimodal",
           "Time_varying_dispersion": "Time-varying dispersion", "Time_varying_markov_switch": "Time-varying Markov switch"}
METHODS = ("rg", "rc", "bg", "bc")
LABELS = {"rg": "RQS-GAS", "rc": "RQS-Con", "bg": "Beta-GAS", "bc": "Beta-Con"}
# static expanding-window maps (eq. 9): key -> (label, family whose colour they share)
STATIC = {"rs": ("Static RQS", "rg"), "bs": ("Static Beta", "bg")}
STATIC_SRC = {"rs": "rg", "bs": "bg"}           # key in r["static_gain"]
#COLORS = {"rg": "#2a78d6", "rc": "#f01708", "bg": "#1baf7a", "bc": "#f5e916", "raw": "#52514e"}
EDGE = {"bc": "#9a9000"}                      # yellow needs an outline to be visible
TESTS = [("KS_Test", "KS"), ("AD_Test", "AD"), ("Chi2_Test", r"$\chi^2$"),
         ("DH_Test", "DH"), ("LB_Test", "LB"), ("Berkowitz_Test", "Berk.")]
INK = "#0b0b0b"


# ===================================================================== DGPs (copied from the notebook)
def _ar_x(v_t):
    x = np.empty(T); x[0] = SIGMA_X * v_t[0]
    for i in range(1, T):
        x[i] = PHI * x[i - 1] + SIGMA_X * v_t[i]
    return x


def generate_levels(p_stay=0.99):
    transitions = np.asarray([-1, 0, 1])
    q = (1 - p_stay) / 2
    tm = np.array([[p_stay, q, q], [q, p_stay, q], [q, q, p_stay]])
    lv = [0]
    for _ in range(1, T):
        lv.append(np.random.choice(transitions, p=tm[np.where(transitions == lv[-1])[0][0]]))
    return np.asarray(lv)


def generate_levels_markov():
    transitions = np.asarray([-1, 0, 1])
    tm1 = np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005], [0.005, 0.005, 0.99]])
    tm2 = np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]])
    lv = [0]
    for i in range(1, T):
        tm = tm1 if i < 1000 else tm2
        lv.append(np.random.choice(transitions, replace=True,
                                   p=tm[np.where(transitions == lv[-1])[0][0]]))
    return np.asarray(lv)


def simulate(name, seed, levels=None, return_x=False):
    """Forecasts (K x T DataFrame), y, regime S_t (TV dispersion only, else None), levels
    [, x if return_x -- the latent AR state, needed for the true conditional density].
    Same draws in the same order as the notebook's get_forecasts / get_forecasts_markov."""
    out = _simulate(name, seed, levels)
    return out if return_x else out[:4]


def true_logpdf(name, x, levels, y):
    """Log density of y_t under the DGP given (x_t, level_t): the oracle forecast. Its log
    score minus the raw pool's is the largest gain ANY forecast could reach (an upper bound
    for recalibration, which cannot add the information the pool lacks)."""
    scen = SCENARIOS[name]
    m = PHI * np.asarray(x, float) + np.asarray(levels, float)
    y = np.asarray(y, float)
    if scen == "under_dispersed":
        return norm.logpdf(y, m, 1.5 * SIGMA_Y)
    if scen == "bimodal":
        return np.log(np.maximum(0.5 * norm.pdf(y, m - 1, SIGMA_Y) + 0.5 * norm.pdf(y, m + 1, SIGMA_Y), 1e-300))
    if scen == "trimodal":
        dens = sum(norm.pdf(y, m + c, 0.15) for c in (-1.5, 0.0, 1.5)) / 3
        return np.log(np.maximum(dens, 1e-300))
    return norm.logpdf(y, m, SIGMA_Y)          # original, markov, biased, TV dispersion


def _simulate(name, seed, levels=None):
    scen = SCENARIOS[name]
    saved = np.random.get_state()
    np.random.seed(seed)
    try:
        state = None
        if scen == "markov":
            lv = generate_levels_markov().astype(float)
            v_t, e_t = norm.rvs(size=T), norm.rvs(size=T)
            x = _ar_x(v_t)
            y = PHI * x + e_t * SIGMA_Y + lv
            z = x + norm.rvs(size=(K, T)) * SIGMA_Z
            p = (z.T + np.linspace(-2, 2, K)).T
            return pd.DataFrame(p), y, None, lv, x

        lv = np.asarray(FIXED_LEVELS if levels is None else levels, float)[:T]
        v_t, e_t = norm.rvs(size=T), norm.rvs(size=T)
        x = _ar_x(v_t)
        y = PHI * x + e_t * SIGMA_Y
        if scen == "bimodal":
            y = y + np.where(np.random.rand(T) < 0.5, -1.0, 1.0)
        if scen == "trimodal":
            y = PHI * x + np.random.choice([-1.5, 0.0, 1.5], size=T) + e_t * 0.15
        if scen == "under_dispersed":
            y = PHI * x + e_t * SIGMA_Y * 1.5
        y = y + lv

        if scen == "original":
            z = x + norm.rvs(size=(K, T)) * SIGMA_Z
            p = (z.T + np.linspace(-2, 2, K)).T
            return pd.DataFrame(p), y, None, lv, x

        spread = np.full(T, SIGMA_Y)
        bias = np.zeros(K)
        if scen == "under_dispersed": spread = spread * 0.9
        if scen == "over_dispersed":  spread = spread * 2.5
        if scen == "biased":          bias = bias + 0.2
        if scen == "bimodal":         spread = np.full(T, np.sqrt(SIGMA_Y**2 + 1))
        if scen == "trimodal":        spread = np.full(T, np.sqrt(0.15**2 + 1.5))
        if scen == "time_varying_dispersion":
            state = generate_levels(p_stay=0.99) != 0
            spread = SIGMA_Y * np.where(state, 0.05, 5)
        p = PHI * x + lv + spread * norm.rvs(size=(K, T)) + bias[:, None]
        return pd.DataFrame(p), y, state, lv, x
    finally:
        np.random.set_state(saved)


def _ci(v, level=0.95):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return (v.mean() if n else np.nan), np.nan, np.nan, n
    se = v.std(ddof=1) / np.sqrt(n)
    return v.mean(), se, student_t.ppf(0.5 + level / 2, n - 1) * se, n



ORDER = ["raw", "rg", "rc", "bg", "bc"]
ROW_LABELS = ["Raw pool", "RQS-GAS", "RQS-Con", "Beta-GAS", "Beta-Con"]
ORDER_S = ["raw", "rs", "rg", "rc", "bs", "bg", "bc"]
ROW_LABELS_S = ["Raw pool", "Static RQS", "RQS-GAS", "RQS-Con", "Static Beta", "Beta-GAS", "Beta-Con"]
F32 = np.float32


# ===================================================================== one job
def run_one(name, seed, warmup=500, eval_start=500, alpha=0.95, loc_deltas=(0.9,),
            full=False, full_deltas=(0.9, 0.2), M=1001, c=1e-20, con_kw=None,
            K_rqs=6, step_norm="coord", slow_shrink=1.0, floor=1e-3,
            static_refit_every=None, roll_windows=(250, 500), stat_deltas=(0.9, 0.2),
            ablations=True):
    """One (scenario, seed) at ALL 12 discount factors. Keeps PITs, per-period gains, test
    flags and the localized metrics at loc_deltas. full=True also keeps the four full
    reports at full_deltas (for the Beta-parameter and other single-seed figures).

    Review additions (one configuration for every method):
      K_rqs, step_norm, slow_shrink, floor   shared by all four methods and the benchmarks
      static_refit_every     refit cadence of the static baselines (default: the network's)
      roll_windows           rolling-window static baselines, at stat_deltas
      stat_deltas            deltas with full per-model statistics (out['stats'])
      ablations              network-only / linear head / no median_loc, at stat_deltas
    Also stores the static expanding map's gain at all 12 deltas (out['static_gain']) and
    the oracle gap log g_t(y_t) - log f_t(y_t) under the true DGP (out['oracle_gap'])."""
    import torch
    torch.set_num_threads(1)
    from ldf import ldf_func
    from pit import pit_func
    from utils.utils import tlp
    from Log_likelihood import tlp_pdf
    from uniformity_tests import uniformity_test
    from localised_metrics import evaluate_localized, static_reports
    import sim_report as sr
    from recal_dma_new import tvgas_pit_dma_new, convolution_recal_pit_dma_new
    from recal_beta_dma_new import tvgas_pit_beta_dma_new, convolution_recal_pit_beta_dma_new

    forecasts, y, state, levels, x_lat = simulate(name, seed, return_x=True)
    f = forecasts.to_numpy()
    ll = np.log(norm.pdf(y, loc=f, scale=SIGMA_Y))
    res = ldf_func(ll, levels=2, discount_factors=[DISCOUNT_FACTORS, DISCOUNT_FACTORS],
                   activation_functions=["softmax", "softmax"], c=c)
    x_grid = np.linspace(f.min() - 3 * SIGMA_Y, f.max() + 3 * SIGMA_Y, M)
    pool = tlp(forecasts, x_grid, res["weights"], SIGMA_Y)
    pit = pit_func(pool, x_grid, y)
    raw_ls = np.log(tlp_pdf(forecasts, x_grid, res["weights"], y, SIGMA_Y))

    con = dict(refit_every=50, hidden_size=4, num_epochs=300, learning_rate=1e-3, seed=42)
    con.update(con_kw or {})
    s_refit = con["refit_every"] if static_refit_every is None else static_refit_every
    common = dict(x=x_grid, y=y, pit_data=pit, cdf_data=pool, discount_factors=DISCOUNT_FACTORS,
                  warmup=warmup, alpha=alpha, floor=floor, eval_start=eval_start)
    gas_kw = dict(static_baseline=True, static_refit_every=s_refit)
    reps = {"rg": tvgas_pit_dma_new(K=K_rqs, verbose=False, **gas_kw, **common),
            "rc": convolution_recal_pit_dma_new(K=K_rqs, verbose=False, **con, **common),
            "bg": tvgas_pit_beta_dma_new(verbose=False, **gas_kw, **common),
            "bc": convolution_recal_pit_beta_dma_new(verbose=False, **con, **common)}

    nd = len(DISCOUNT_FACTORS)
    pits = {"raw": pit[:, eval_start:]}
    pits.update({m: np.vstack([reps[m][i]["calibrated_pits"] for i in range(nd)]) for m in METHODS})
    pits["rs"] = np.vstack([np.asarray(reps["rg"][i]["v_static"])[eval_start:] for i in range(nd)])
    pits["bs"] = np.vstack([np.asarray(reps["bg"][i]["v_static"])[eval_start:] for i in range(nd)])
    passed = {}
    for k, p in pits.items():
        _, summ = uniformity_test(p, DISCOUNT_FACTORS)
        passed[k] = np.array([[str(summ[col].iloc[i]).strip().upper() == "PASS" for col, _ in TESTS]
                              for i in range(nd)])

    tmp = {}
    sr.add_scenario(tmp, name, **{k: pits[k] for k in ORDER})
    for d in loc_deltas:
        loc = evaluate_localized(x=x_grid, y=y, pit_data=pit, cdf_data=pool,
                                 methods={"GAS spline": reps["rg"], "CONV spline": reps["rc"],
                                          "GAS beta": reps["bg"], "CONV beta": reps["bc"],
                                          "Static spline": static_reports(reps["rg"], "rqs", K=K_rqs),
                                          "Static beta": static_reports(reps["bg"], "beta")},
                                 discount_factor=float(d), eval_start=eval_start,
                                 min_events=55, n_blocks=4, verbose=False)
        sr.add_localized(tmp, name, loc)

    out = dict(name=name, seed=seed, eval_start=eval_start, y=y.astype(F32), state=state,
               pits={k: v.astype(F32) for k, v in pits.items()},
               gain={m: np.vstack([reps[m][i]["log_likelihood"] for i in range(nd)]).astype(F32)
                     for m in METHODS},
               raw_logscore=raw_ls.astype(F32),
               ks={k: np.array([kstest(v[i], "uniform").statistic for i in range(nd)]) for k, v in pits.items()},
               passed=passed, localized=tmp[name]["localized"],
               levels=np.asarray(levels, float).astype(F32))
    # where Con's gain comes from: total = network layer + outer GAS layer (on the transformed PIT);
    # the GAS-only method is the same bank on the raw PIT (ll_gas_only_raw), so
    #   Con - GAS = mean(ll_nn) + mean(ll_gas - ll_gas_only_raw)
    out["decomp"] = {m: {k: np.vstack([reps[m][i][k] for i in range(nd)]).astype(F32)
                         for k in ("ll_nn", "ll_gas", "ll_gas_only_raw")}
                     for m in ("rc", "bc") if all(k in reps[m][0] for k in ("ll_nn", "ll_gas", "ll_gas_only_raw"))}
    # the map actually applied, summarised per period: mean and SD of tau_t on the PIT scale
    # (identity: 0.5 and 0.289) -- comparable across Beta and RQS and across all scenarios
    out["map_mean"], out["map_sd"] = {}, {}
    for m in METHODS:
        if "tau_grid" in reps[m][0]:
            mm = [_map_moments(reps[m][i]["tau_grid"], reps[m][i]["u_grid"]) for i in range(nd)]
            out["map_mean"][m] = np.vstack([a for a, _ in mm]).astype(F32)
            out["map_sd"][m] = np.sqrt(np.vstack([b for _, b in mm])).astype(F32)
    if full:
        idx = [int(np.flatnonzero(np.isclose(DISCOUNT_FACTORS, d))[0]) for d in full_deltas]
        out["full"] = {m: {float(DISCOUNT_FACTORS[i]): reps[m][i] for i in idx} for m in METHODS}
        out["forecasts"] = forecasts

    # ---------------- review additions: benchmarks, ablations, sharpness, oracle -------------- #
    out["static_gain"] = {"rg": np.vstack([reps["rg"][i]["ll_static"] for i in range(nd)]).astype(F32),
                          "bg": np.vstack([reps["bg"][i]["ll_static"] for i in range(nd)]).astype(F32)}
    # (nd, T): raw_ls is per discount factor
    out["oracle_gap"] = (true_logpdf(name, x_lat, levels, y)[None, :] - raw_ls).astype(F32)
    out["stats"] = {float(d): _stats_at(d, reps, pit, pool, x_grid, y, eval_start, warmup, K_rqs,
                                        s_refit, roll_windows, ablations, con, common)
                    for d in stat_deltas}
    out["settings"] = dict(K_rqs=K_rqs, step_norm=step_norm, slow_shrink=slow_shrink, alpha=alpha,
                           floor=floor, warmup=warmup, eval_start=eval_start, static_refit_every=s_refit,
                           roll_windows=tuple(roll_windows), **{k: v for k, v in con.items()})
    return out


# ===================================================================== review additions
BENCH_KEYS = ["s_rqs_exp", "s_beta_exp"]
ABLATIONS = {"nn_only": dict(outer="none"), "linear": dict(hidden_size=0), "no_loc": "no_loc"}
MODEL_LABELS = {"raw": "Raw pool", "s_rqs_exp": "Static-RQS (exp)", "s_beta_exp": "Static-Beta (exp)",
                **{m: LABELS[m] for m in METHODS},
                **{f"{fam}:{a}": f"{lab}: {txt}" for fam, lab in (("rc", "RQS-Con"), ("bc", "Beta-Con"))
                   for a, txt in (("nn_only", "network only"), ("linear", "linear head"),
                                  ("no_loc", "no median_loc"))}}


def _roll_key(fam, w):
    return f"s_{fam}_roll{w}"


def _model_label(k):
    if k in MODEL_LABELS:
        return MODEL_LABELS[k]
    if k.startswith("s_") and "_roll" in k:
        fam, w = k[2:].split("_roll")
        return f"Static-{'RQS' if fam == 'rqs' else 'Beta'} (roll {w})"
    return k


def raw_moment_test(v, n_mom=4):
    """H0: E[v^k] = 1/(k+1), k = 1..n_mom (uniform PIT), with a Newey-West covariance, so it
    stays valid under serial dependence (in the spirit of Knuppel 2015). Returns (stat, p)."""
    from recal_dma_new import _hac_lag
    from scipy.stats import chi2
    v = np.asarray(v, float)
    m = np.column_stack([v ** k - 1.0 / (k + 1) for k in range(1, n_mom + 1)])
    n = len(v); mbar = m.mean(0); e = m - mbar; L = _hac_lag(n)
    S = e.T @ e / n
    for lag in range(1, L + 1):
        G = e[lag:].T @ e[:-lag] / n
        S += (1 - lag / (L + 1)) * (G + G.T)
    stat = float(n * mbar @ np.linalg.solve(S, mbar))
    return stat, float(chi2.sf(stat, n_mom))


def _stats_at(delta, reps, pit, pool, x_grid, y, es, warmup, K_rqs, s_refit, roll_windows,
              ablations, con, common):
    """Per-model statistics at one delta on [es, T): per-period gain over the raw pool (full
    length, F32), mean central-interval width / coverage / score, moment-test p and KS."""
    from recal_dma_new import (run_static_expanding, _prep_pit, _spline_on_grid, _pit_moments,
                               _pit_quantiles, _UN, _IVL_PROBS, _interval_metrics, FEATURE_NAMES,
                               convolution_recal_pit_dma_new)
    from recal_beta_dma_new import (run_static_expanding_beta, _bank_eval_beta,
                                    convolution_recal_pit_beta_dma_new)
    i = _didx(delta)
    Tn = len(y)
    sl = slice(es, Tn)
    u = _prep_pit(pit[i])
    q_raw = np.column_stack([reps["rg"][i]["y_raw_q05"], reps["rg"][i]["y_raw_q95"]])
    crps_raw = reps["rg"][i]["crps_raw_history"]

    def entry(ll, v_eval, q, crps):
        ivl = _interval_metrics(q, y, sl)
        return dict(ll=np.asarray(ll, F32), width=ivl["width"], coverage=ivl["coverage"],
                    score=ivl["score"], crps=float(np.mean(crps[sl])),
                    moment_p=raw_moment_test(v_eval)[1], ks=float(kstest(v_eval, "uniform").statistic))

    def static_q(kind, path):
        q = np.zeros((Tn, 2)); cr = np.zeros(Tn)
        for t in range(Tn):
            if kind == "rqs":
                tau, dtau = _spline_on_grid(path[t], K_rqs)
            else:
                tau, dtau = _bank_eval_beta(path[t][None, :], _UN, np.array([1.0]))
            _, cr[t] = _pit_moments(pool[i][t], x_grid, y[t], tau, dtau)
            q[t] = _pit_quantiles(pool[i][t], x_grid, tau, _IVL_PROBS)
        return q, cr

    res = {"raw": entry(np.zeros(Tn), u[sl], q_raw, crps_raw)}
    for m in METHODS:
        r = reps[m][i]
        res[m] = entry(r["log_likelihood"], r["calibrated_pits"],
                       np.column_stack([r["y_calibrated_q05"], r["y_calibrated_q95"]]), r["crps_history"])
    # static baselines: expanding (from the GAS runs) and rolling
    for key, kind, src in (("s_rqs_exp", "rqs", reps["rg"][i]["theta_static"]),
                           ("s_beta_exp", "beta", reps["bg"][i]["logits_static"])):
        ll = (reps["rg"] if kind == "rqs" else reps["bg"])[i]["ll_static"]
        v = (reps["rg"] if kind == "rqs" else reps["bg"])[i]["v_static"]
        q, cr = static_q(kind, src)
        res[key] = entry(ll, v[sl], q, cr)
    for w in roll_windows:
        ll, path, v = run_static_expanding(u, K_rqs, warmup, refit_every=s_refit, window=w)
        q, cr = static_q("rqs", path)
        res[_roll_key("rqs", w)] = entry(ll, v[sl], q, cr)
        ll, path, v = run_static_expanding_beta(u, warmup, refit_every=s_refit, window=w)
        q, cr = static_q("beta", path)
        res[_roll_key("beta", w)] = entry(ll, v[sl], q, cr)
    # ablations of the composition
    if ablations:
        one = dict(common, pit_data=pit[i:i + 1], cdf_data=pool[i:i + 1],
                   discount_factors=DISCOUNT_FACTORS[i:i + 1])
        no_loc = [f for f in FEATURE_NAMES if f != "median_loc"]
        for a, extra in ABLATIONS.items():
            extra = dict(features=no_loc) if extra == "no_loc" else extra
            kw = {**con, **extra}
            for fam, fn in (("rc", lambda **k: convolution_recal_pit_dma_new(K=K_rqs, **k)),
                            ("bc", convolution_recal_pit_beta_dma_new)):
                r = fn(verbose=False, **kw, **one)[0]
                res[f"{fam}:{a}"] = entry(r["log_likelihood"], r["calibrated_pits"],
                                          np.column_stack([r["y_calibrated_q05"], r["y_calibrated_q95"]]),
                                          r["crps_history"])
    return res


def _hac_t(d):
    from recal_dma_new import dm_test, _hac_lag
    d = np.asarray(d, float)
    return dm_test(d, hac_lag=_hac_lag(len(d)))[2]


def benchmark_table(reps, delta=0.9, names=None, models=None):
    """Per scenario and model, across seeds: mean log-score gain over the raw pool and its
    across-seed t; share of seeds with a significant HAC DM gain / loss (5%); mean coverage and
    width (relative to raw) of the central interval; interval-score and CRPS change in %;
    share of seeds passing the moment test at 5%; share of the oracle gap captured."""
    names = _names(reps) if names is None else names
    key = float(delta)
    rows = []
    for n in names:
        rs = [r for r in reps if r["name"] == n]
        if key not in rs[0].get("stats", {}):
            raise ValueError(f"no stats at delta={delta}; rerun with stat_deltas including it")
        mods = models or list(rs[0]["stats"][key])
        for mkey in mods:
            g, tv, cov, wid, isc, crp, mp, orc = [], [], [], [], [], [], [], []
            for r in rs:
                s = r["stats"][key]
                es_ = r["eval_start"]
                d = s[mkey]["ll"][es_:].astype(float)
                g.append(d.mean())
                tv.append(_hac_t(d) if mkey != "raw" else np.nan)
                cov.append(s[mkey]["coverage"])
                wid.append(s[mkey]["width"] / s["raw"]["width"])
                isc.append(100 * (1 - s[mkey]["score"] / s["raw"]["score"]))
                crp.append(100 * (1 - s[mkey]["crps"] / s["raw"]["crps"]))
                mp.append(s[mkey]["moment_p"] > 0.05)
                orc.append(r["oracle_gap"][_didx(delta)][es_:].astype(float).mean())
            g, tv = np.asarray(g), np.asarray(tv)
            mean, se, _, k = _ci(g)
            rows.append({"scenario": n, "model": _model_label(mkey), "key": mkey, "R": k,
                         "dLogS": mean, "t (seeds)": mean / se if se else np.nan,
                         "sig+ share": np.mean(tv > 1.96) if mkey != "raw" else np.nan,
                         "sig- share": np.mean(tv < -1.96) if mkey != "raw" else np.nan,
                         "dCRPS %": np.mean(crp), "dIS %": np.mean(isc),
                         "coverage": np.mean(cov), "width/raw": np.mean(wid),
                         "moment pass": np.mean(mp),
                         "oracle share %": 100 * mean / np.mean(orc) if np.mean(orc) > 0 else np.nan})
    return pd.DataFrame(rows)


def pairwise_table(reps, pairs, delta=0.9, names=None):
    """Across-seed comparison of model pairs at one delta: mean difference in log-score gain,
    across-seed t, share of seeds with a significant HAC DM difference either way, wins."""
    names = _names(reps) if names is None else names
    key = float(delta)
    rows = []
    for n in names:
        rs = [r for r in reps if r["name"] == n]
        for a, b in pairs:
            d_means, tv = [], []
            for r in rs:
                s = r["stats"][key]; es_ = r["eval_start"]
                d = (s[a]["ll"] - s[b]["ll"])[es_:].astype(float)
                d_means.append(d.mean()); tv.append(_hac_t(d))
            d_means, tv = np.asarray(d_means), np.asarray(tv)
            mean, se, _, k = _ci(d_means)
            rows.append({"scenario": n, "comparison": f"{_model_label(a)} - {_model_label(b)}",
                         "R": k, "mean diff": mean, "t (seeds)": mean / se if se else np.nan,
                         "sig+ share": np.mean(tv > 1.96), "sig- share": np.mean(tv < -1.96),
                         "wins": f"{int((d_means > 0).sum())}/{k}"})
    return pd.DataFrame(rows)


def time_variation_table(reps, names=None):
    """GAS minus the static expanding map of the same family, at EVERY discount factor:
    mean difference across seeds and across-seed t. The test of whether time variation helps."""
    names = _names(reps) if names is None else names
    rows = []
    for n in names:
        rs = [r for r in reps if r["name"] == n]
        for fam, m in (("RQS", "rg"), ("Beta", "bg")):
            for i, dfac in enumerate(DISCOUNT_FACTORS):
                d = np.asarray([(r["gain"][m][i] - r["static_gain"][m][i])[r["eval_start"]:].astype(float).mean()
                                for r in rs])
                mean, se, _, k = _ci(d)
                rows.append({"scenario": n, "family": fam, "delta": dfac, "R": k,
                             "GAS - static": mean, "t (seeds)": mean / se if se else np.nan,
                             "wins": int((d > 0).sum())})
    return pd.DataFrame(rows)


def benchmark_latex(reps, delta=0.9, names=None, models=None, label="tab:sim_benchmarks", caption=None):
    """LaTeX: per scenario and model, mean log-score gain (stars: across-seed t-test), share of
    seeds with a significant HAC gain, coverage, width relative to raw, moment-test pass rate,
    share of the oracle gap captured."""
    tab = benchmark_table(reps, delta, names, models)
    R = int(tab["R"].max())
    lines = [r"\begin{table}[H]\centering\small",
             r"\begin{tabular}{ll r c c c c r}\toprule",
             r"Scenario & Model & $\Delta$LogS & Sig.\ gain & Cov. & Width/raw & Moment pass & Oracle \% \\ \midrule"]
    for sc, g in tab.groupby("scenario", sort=False):
        first = True
        for _, row in g.iterrows():
            s = _stars(row["t (seeds)"], int(row.R))
            gain = f"${row.dLogS:+.4f}" + (f"^{{{s}}}" if s else "") + "$"
            sig = "--" if row.key == "raw" else f"{row['sig+ share']:.1f}"
            orc = "--" if not np.isfinite(row["oracle share %"]) or row.key == "raw" else f"{row['oracle share %']:.0f}"
            lines.append(f"{DISPLAY.get(sc, sc) if first else ''} & {row.model} & {gain} & {sig} & "
                         f"{row.coverage:.2f} & {row['width/raw']:.2f} & {row['moment pass']:.1f} & {orc} \\\\")
            first = False
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule\end{tabular}"
    cap = caption or (rf"Simulation benchmarks at $\delta={delta}$, {R} replications. $\Delta$LogS: mean log-score "
                      r"gain over the raw pool, stars from the across-replication $t$-test (10/5/1\%). Sig.\ gain: "
                      r"share of replications with a HAC Diebold--Mariano gain significant at 5\%. Cov.\ and "
                      r"Width/raw: coverage and mean width (relative to the raw pool) of the central 90\% interval. "
                      r"Moment pass: share of replications not rejecting the dependence-robust raw-moment "
                      r"calibration test at 5\%. Oracle \%: mean gain as a share of the log-score gap between the "
                      r"true DGP density and the raw pool.")
    lines += [rf"\caption{{{cap}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def _map_moments(tau_grid, u_grid):
    """Mean and variance of the map tau_t read as a CDF on (0, 1), Gauss-Legendre quadrature on
    the stored nodes (checked), trapezoid otherwise."""
    tau = np.asarray(tau_grid, float); u = np.asarray(u_grid, float)
    x, w = np.polynomial.legendre.leggauss(len(u))
    if np.allclose(0.5 * (x + 1.0), u, atol=1e-10):
        w = 0.5 * w
        m = 1.0 - tau @ w
        s2 = 1.0 - 2.0 * tau @ (u * w)
    else:
        uu = np.r_[0.0, u, 1.0]
        tt = np.concatenate([np.zeros((len(tau), 1)), tau, np.ones((len(tau), 1))], axis=1)
        m = 1.0 - np.trapezoid(tt, uu, axis=1)
        s2 = 1.0 - 2.0 * np.trapezoid(tt * uu, uu, axis=1)
    return m, np.maximum(s2 - m ** 2, 1e-12)


def run_all(seeds, scenarios=None, n_jobs=4, path=None, full_seed=None,
            full_scenarios=("Time_varying_dispersion",), verbose=10, **kw):
    """Every (scenario, seed). full_seed (default: the first seed) keeps full reports for
    full_scenarios, for the single-seed figures."""
    from joblib import Parallel, delayed
    seeds = list(seeds)
    full_seed = seeds[0] if full_seed is None else full_seed
    names = list(SCENARIOS) if scenarios is None else list(scenarios)
    jobs = [(n, s) for s in seeds for n in names]
    reps = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_one)(n, s, full=(s == full_seed and n in full_scenarios), **kw) for n, s in jobs)
    if path:
        save(reps, path)
    return reps


def save(reps, path):
    with open(path, "wb") as fh:
        pickle.dump(reps, fh)


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _didx(delta):
    return int(np.flatnonzero(np.isclose(DISCOUNT_FACTORS, delta))[0])


def _names(reps):
    seen = {r["name"] for r in reps}
    return [n for n in SCENARIOS if n in seen]


def _get(reps, name, seed):
    for r in reps:
        if r["name"] == name and r["seed"] == seed:
            return r
    raise KeyError((name, seed))


# ===================================================================== representative seed -> sim_report
def sr_results(reps, seed, names=None):
    """{scenario: entry} for ONE seed in sim_report's format, so raw_hist_all, recal_hist_all,
    ecdf_dev_all, pit_grid and pit_panel work unchanged."""
    names = _names(reps) if names is None else names
    out = {}
    for n in names:
        r = _get(reps, n, seed)
        e = {"group": "", "localized": r["localized"]}
        e.update({k: {"pits": np.asarray(r["pits"][k], float)} for k in ORDER + ["rs", "bs"] if k in r["pits"]})
        out[n] = e
    return out


# ===================================================================== replicated test heatmap
def pass_means(reps, names=None, order=None):
    """{display name: rows x 6} mean (over seeds) number of the 12 discount factors passing each
    test. Uses the static rows when the reps contain them, else the original five."""
    names = _names(reps) if names is None else names
    if order is None:
        order = ORDER_S if all(k in reps[0]["passed"] for k in ("rs", "bs")) else ORDER
    out, R = {}, {}
    for n in names:
        rs = [r for r in reps if r["name"] == n]
        mats = np.stack([np.stack([r["passed"][k].sum(axis=0) for k in order]) for r in rs])
        out[DISPLAY.get(n, n)] = mats.mean(axis=0)
        R[DISPLAY.get(n, n)] = len(rs)
    return out, R, order


def plot_pass_heatmaps(reps, names=None, path=None, ncol=4, cmap="Blues", panel=(3.5, 3.6),
                       decimals=1, fontsize=11):
    """Test-heatmap grid, each cell the MEAN over seeds of the number of the 12 discount factors
    at which the test does not reject. Rows: raw, then each family's static, GAS and Con."""
    means, Rs, order = pass_means(reps, names)
    row_labels = ROW_LABELS_S if order == ORDER_S else ROW_LABELS
    labs = list(means)
    N = len(DISCOUNT_FACTORS)
    nrow = int(np.ceil(len(labs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14,7),
                             squeeze=False, layout="constrained")
    for k, (ax, lab) in enumerate(zip(axes.flat, labs)):
        Mx = means[lab]
        im = ax.imshow(Mx, cmap=cmap, vmin=0, vmax=N, aspect="auto")
        for i, j in np.ndindex(Mx.shape):
            ax.text(j, i, f"{Mx[i, j]:.{decimals}f}", ha="center", va="center", fontsize=fontsize,
                    color="white" if Mx[i, j] >= 2 * N / 3 else "#2b2b2b")
        ax.axvline(3.5, color="white", lw=1.5)
        ax.axhline(0.5, color="white", lw=1.5)                  # raw | methods
        if order == ORDER_S:
            ax.axhline(3.5, color="white", lw=1.5)              # RQS block | Beta block
        ax.set_xticks(range(len(TESTS)), [h for _, h in TESTS], fontsize=fontsize,color=INK)
        ax.set_yticks(range(len(order)), row_labels if k % ncol == 0 else [], fontsize=fontsize,color=INK)
        ax.tick_params(length=0,color=INK); ax.grid(False)
        ax.set_title(lab, fontsize=fontsize + 2)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes.flat[len(labs):]:
        ax.axis("off")
    R = max(Rs.values())
    if len(labs) < axes.size:
        cb = fig.colorbar(im, cax=axes.flat[len(labs)].inset_axes([0.1, 0.1, 0.07, 0.8]))
    else:
        cb = fig.colorbar(im, ax=list(axes.flat), shrink=0.9)
    cb.set_label(f"Non-rejections (of {N})" + (f", mean over {R} seeds" if R > 1 else ""), fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    #return fig


# ===================================================================== replicated scores table
SCORE_ROWS = [("logs", "LogS", +1), ("crps", "CRPS", -1), ("is90", r"IS$_{90}$", -1),
              ("is95", r"IS$_{95}$", -1), ("csl_l", r"CSL$_{L}$", +1), ("csl_r", r"CSL$_{R}$", +1)]


# static maps in the scorecard: key -> its entry in r["stats"][delta]. The localized metrics
# (IS95, CSL) exist only for the four methods; for the static maps KS, LogS, CRPS and IS90 come
# from r["stats"], with the gain taken against the raw pool from the same source.
STATIC_STATS = {"rs": "s_rqs_exp", "bs": "s_beta_exp"}
SCORE_METHODS = ("rs", "rg", "rc", "bs", "bg", "bc")
SCORE_LABELS = {**LABELS, "rs": "Static RQS", "bs": "Static Beta"}


def _static_rows(r, delta, names_ok=True):
    """Scorecard rows of the static maps for one replication, from r['stats'][delta]."""
    from scipy.stats import kstwo
    s = r.get("stats", {}).get(float(delta))
    if s is None:
        return []
    es_ = r["eval_start"]
    n = len(r["y"]) - es_
    raw = s["raw"]
    rows = []
    for m, key in STATIC_STATS.items():
        if key not in s:
            continue
        e = s[key]
        vals = {"ks": (raw["ks"], e["ks"], -1),
                "logs": (0.0, float(np.asarray(e["ll"], float)[es_:].mean()), +1),
                "crps": (raw["crps"], e["crps"], -1),
                "is90": (raw["score"], e["score"], -1)}
        for k, (rv, lev, sign) in vals.items():
            lab = "KS" if k == "ks" else dict((a, b) for a, b, _ in SCORE_ROWS)[k]
            gain = (lev - rv) if k == "logs" else sign * (lev - rv)
            rows.append(dict(scenario=r["name"], seed=r["seed"], metric=lab, key=k, method=m,
                             raw=np.nan, level=lev, gain=gain,
                             ks_p=float(kstwo.sf(e["ks"], n)) if k == "ks" else np.nan,
                             raw_ks_p=np.nan))
    return rows


def scores_frame(reps, delta=0.9, names=None, static=True):
    """Long table: one row per (scenario, seed, metric, method) with the level and the gain
    over the raw pool (oriented so + = better). static=True adds the static maps ('rs', 'bs')
    for KS, LogS, CRPS and IS90, from r['stats'][delta]."""
    names = _names(reps) if names is None else names
    key = round(float(delta), 6)
    rows = []
    for r in reps:
        if r["name"] not in names:
            continue
        loc = r["localized"].get(key)
        if loc is None:
            raise ValueError(f"{r['name']} seed {r['seed']}: no localized metrics at delta={delta}; "
                             f"rerun with loc_deltas including it")
        have_static = all(m in loc.get("logs", {}).get("level", {}) for m in STATIC_STATS)
        meths = SCORE_METHODS if (static and have_static) else METHODS
        for k, lab, sign in [("ks", "KS", -1)] + SCORE_ROWS:
            if k not in loc:
                continue
            raw = loc[k]["raw"]
            for m in meths:
                lev = loc[k]["level"][m]
                rows.append(dict(scenario=r["name"], seed=r["seed"], metric=lab, key=k,
                                 method=m, raw=raw, level=lev, gain=sign * (lev - raw),
                                 ks_p=loc["ks_p"]["level"][m] if k == "ks" and "ks_p" in loc else np.nan,
                                 raw_ks_p=loc["ks_p"]["raw"] if k == "ks" and "ks_p" in loc else np.nan))
        if static and not have_static:          # older runs: KS/LogS/CRPS/IS90 from r["stats"]
            rows.extend(_static_rows(r, delta))
    return pd.DataFrame(rows)


def _stars(tt, n):
    if not np.isfinite(tt) or n < 2:
        return ""
    p = 2 * student_t.sf(abs(tt), n - 1)
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def scores_table(reps, delta=0.9, names=None, static=True):
    """Across-seed summary of the scorecard: raw level (mean), each method's mean gain with
    its across-seed se and t. KS rows: mean KS and the share of seeds rejecting at 5%.
    static=True adds Static RQS / Static Beta columns (NaN where a metric is unavailable)."""
    df = scores_frame(reps, delta, names, static)
    methods = SCORE_METHODS if static else METHODS
    out = []
    for (sc, metric), g in df.groupby(["scenario", "metric"], sort=False):
        gm_raw = g[g.method.isin(METHODS)]
        row = dict(scenario=sc, metric=metric, R=g["seed"].nunique(),
                   raw=gm_raw.groupby("seed")["raw"].first().mean())
        for m in methods:
            gm = g[g.method == m]
            if gm.empty:
                row[SCORE_LABELS[m]] = np.nan
                row[SCORE_LABELS[m] + (" rej" if metric == "KS" else " t")] = np.nan
                continue
            if metric == "KS":
                row[SCORE_LABELS[m]] = gm["level"].mean()
                row[SCORE_LABELS[m] + " rej"] = (gm["ks_p"] < .05).mean()
            else:
                mean, se, _, n = _ci(gm["gain"])
                row[SCORE_LABELS[m]] = mean
                row[SCORE_LABELS[m] + " t"] = mean / se if se else np.nan
        if metric == "KS":
            row["raw rej"] = (gm_raw.groupby("seed")["raw_ks_p"].first() < .05).mean()
        out.append(row)
    return pd.DataFrame(out)


def scores_latex(reps, delta=0.9, names=None, label="tab:sim_scorecard_rep", caption=None,
                 static=True):
    """LaTeX version of the scorecard averaged over seeds. Method columns: mean gain over the
    raw pool across seeds, stars from the across-seed t-test; KS rows: mean KS with the share
    of seeds rejecting uniformity at 5% in brackets. Best method per row in bold.
    static=True adds Static RQS and Static Beta columns ('--' for IS95 and CSL)."""
    tab = scores_table(reps, delta, names, static)
    meth = [SCORE_LABELS[m] for m in (SCORE_METHODS if static else METHODS)]
    fin = lambda v: v is not None and np.isfinite(v)
    lines = [r"\begin{table}[H]\centering\small",
             r"\begin{tabular}{ll" + "c" * (1 + len(meth)) + r"}\toprule",
             "Scenario & Metric & Raw pool & " + " & ".join(meth) + r" \\ \midrule"]
    R = int(tab["R"].max())
    for sc, g in tab.groupby("scenario", sort=False):
        first = True
        for _, row in g.iterrows():
            if row.metric == "KS":
                best = min(meth, key=lambda m: row[m] if fin(row[m]) else np.inf)
                cells = [f"${row.raw:.3f}$ [{row['raw rej']:.1f}]"] + \
                        [("--" if not fin(row[m]) else
                          (rf"$\mathbf{{{row[m]:.3f}}}$" if m == best else f"${row[m]:.3f}$") + f" [{row[m + ' rej']:.1f}]")
                         for m in meth]
            else:
                best = max(meth, key=lambda m: row[m] if fin(row[m]) else -np.inf)
                cells = [f"${row.raw:.4f}$"]
                for m in meth:
                    if not fin(row[m]):
                        cells.append("--")
                        continue
                    s = _stars(row[m + " t"], int(row.R))
                    body = f"{row[m]:+.4f}" + (f"^{{{s}}}" if s else "")
                    cells.append(rf"$\mathbf{{{body}}}$" if m == best else f"${body}$")
            lines.append(f"{DISPLAY.get(sc, sc) if first else ''} & {row.metric} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule\end{tabular}"
    cap = caption or (rf"Simulation results at $\delta={delta}$, averaged over {R} independent replications. "
                      r"KS: mean Kolmogorov--Smirnov distance of the PIT from uniform, with the share of "
                      r"replications rejecting at 5\% in brackets. Raw pool: mean level of each score. Method "
                      r"columns: mean gain over the raw pool, oriented so that positive is better; stars from "
                      r"the $t$-test of the gain across replications at 10/5/1\%. Best method in bold."
                      + (r" Static RQS, Static Beta: expanding-window static maps of each family." if static else ""))
    lines += [rf"\caption{{{cap}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)



# ===================================================================== Con vs GAS
def con_vs_gas(reps, delta=0.9, names=None):
    """Con - GAS, GAS - static and Con - static mean log-score gain across seeds, per scenario
    and family (static = expanding-window map of the same family), with across-seed t and wins;
    for scenarios with a regime (TV dispersion) also split by S_t."""
    names = _names(reps) if names is None else names
    i = _didx(delta)
    out = []
    for n in names:
        rs = [r for r in reps if r["name"] == n]
        for fam, (gas, con) in {"RQS": ("rg", "rc"), "Beta": ("bg", "bc")}.items():
            src = gas                                    # static map of this family
            parts = {"all": None}
            if rs[0]["state"] is not None:
                parts.update({"S=0": False, "S=1": True})
            for lab, val in parts.items():
                row = dict(scenario=n, family=fam, regime=lab)
                for cname, (a, b) in {"con_minus_gas": (con, gas), "gas_minus_static": (gas, "S"),
                                      "con_minus_static": (con, "S")}.items():
                    d = []
                    for r in rs:
                        es = r["eval_start"]
                        bb = r["static_gain"][src][i] if b == "S" else r["gain"][b][i]
                        diff = (r["gain"][a][i] - bb)[es:].astype(float)
                        if val is not None:
                            diff = diff[np.asarray(r["state"])[es:] == val]
                        d.append(diff.mean())
                    d = np.asarray(d)
                    mean, se, _, k = _ci(d)
                    tt = mean / se if se else np.nan
                    row.update({"R": k, cname: mean, f"t {cname}": tt,
                                f"wins {cname}": f"{int((d > 0).sum())}/{k}"})
                out.append(row)
    return pd.DataFrame(out)


# ===================================================================== TVD main-text figures
def plot_cum_gain_avg(reps, name="Time_varying_dispersion", delta=0.9, path=None, level=0.95,
                      difference=False, show_static=True, figsize=(11, 4)):
    """Mean over seeds of the cumulative log-score gain over the raw pool, with a `level` band
    across seeds; show_static adds the static expanding maps (dotted, family colour).
    difference=True plots Con - GAS for each family instead (above 0: Con ahead)."""
    i = _didx(delta)
    rs = [r for r in reps if r["name"] == name]
    es = rs[0]["eval_start"]
    t = np.arange(es, len(rs[0]["y"]))
    fig, ax = plt.subplots(figsize=figsize)
    G = lambda r, m: (r["static_gain"][STATIC_SRC[m]][i] if m in STATIC else r["gain"][m][i])
    if difference:
        series = {"RQS: Con $-$ GAS": ("rc", "rg", COLORS["rc"]), "Beta: Con $-$ GAS": ("bc", "bg", "#1baf7a")}
    else:
        series = {LABELS[m]: (m, None, COLORS[m]) for m in METHODS}
        if show_static:
            series.update({lab: (k, None, COLORS[fam]) for k, (lab, fam) in STATIC.items()})
    for lab, (a, b, col) in series.items():
        C = np.stack([np.cumsum((G(r, a) - (G(r, b) if b else 0))[es:].astype(float)) for r in rs])
        mean = C.mean(0)
        h = student_t.ppf(0.5 + level / 2, len(rs) - 1) * C.std(0, ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else 0
        ax.fill_between(t, mean - h, mean + h, color=col, alpha=0.18, lw=0)
        ls = ":" if a in STATIC else ("--" if (not difference and a in ("rc", "bc")) else "-")
        ax.plot(t, mean, color=col, lw=2.0 if a in STATIC else 1.8, ls=ls, label=lab)
    ax.axhline(0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("Time"); ax.set_ylabel("Cumulative log-score gain" + (" (Con $-$ GAS)" if difference else ""))
    who = f"seed {rs[0]['seed']}" if len(rs) == 1 else f"mean over {len(rs)} seeds"
    ax.set_title(rf"{DISPLAY.get(name, name)}, $\delta={delta}$, {who}", loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def event_runs(reps, delta=0.9, name="Time_varying_dispersion"):
    """`runs` for switch_event_study.plot_event_study: every seed of one regime scenario,
    including the static expanding maps under 'rs' (RQS) and 'bs' (Beta)."""
    i = _didx(delta)
    return [{"state": r["state"], **{m: r["gain"][m][i].astype(float) for m in METHODS},
             **{k: r["static_gain"][src][i].astype(float) for k, src in STATIC_SRC.items()}}
            for r in reps if r["name"] == name and r["state"] is not None]


def full_reports(reps, name="Time_varying_dispersion", delta=0.9):
    """The four full reports kept for the representative seed: {rg, rc, bg, bc}, plus the
    seed's regime -- for plot_beta_effective and the other single-seed figures."""
    for r in reps:
        if r["name"] == name and "full" in r:
            return {m: r["full"][m][float(delta)] for m in METHODS}, r["state"], r["seed"]
    raise KeyError(f"no full reports for {name}; run_all keeps them for full_seed only")