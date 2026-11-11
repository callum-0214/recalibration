"""
Section 7 summary tables and figure across simulation scenarios.

Every metric in the tables is taken from localised_metrics.evaluate_localized, so all
numbers come from one code path, one evaluation window and one set of HAC tests.

Usage (in the notebook, per scenario):

    import sim_report as sr
    results = sr.load_results("sim_results.pkl")        # {} the first time
    sr.add_scenario(results, "Over-dispersed",
                    raw=raw_pits, rg=rg_calibrated_pits, rc=rc_calibrated_pits,
                    bg=bg_calibrated_pits, bc=bc_calibrated_pits)   # (12, T) PITs, for the figure
    for delta in sr.DELTAS:                              # one call per discount factor you report
        loc = evaluate_localized(x=x_grid, y=sim_y, pit_data=Raw_pit, cdf_data=linear_pool,
                                 methods={'GAS spline': rg, 'CONV spline': rc,
                                          'GAS beta': bg, 'CONV beta': bc},
                                 discount_factor=delta, eval_start=eval_start,
                                 min_events=55, n_blocks=4, verbose=False)
        sr.add_localized(results, "Over-dispersed", loc)
    sr.save_results(results, "sim_results.pkl")

Then:

    d = sr.DELTAS.index(0.9)
    print(sr.scorecard_latex(results, d))               # main-text table (needs delta = 0.9 attached)
    print(sr.robustness_latex(results))                 # across every attached delta
    sr.pit_grid(results, d, "fig_sim_pit_grid.pdf")
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DELTAS = [1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.001]
METHODS = {"rg": "RQS-GAS", "rc": "RQS-Con", "bg": "Beta-GAS", "bc": "Beta-Con"}
DAGGER = r"$^\dagger$"

# Categorical slots 1-4 in fixed order (validated: CVD and normal-vision separation pass).
# Yellow/aqua are below 3:1 contrast, so lines also carry style: GAS solid, Con dashed.
COLORS = {"rg": "#2a78d6", "rc": "#f01708", "bg": "#1baf7a", "bc": "#eec037e7", "raw": "#52514e"}
# COLORS={"rg":  "#56B4E9",   # light blue
#         "rc":  "#08306B",   # dark navy
#         "bg": "#F4A259",   # light orange
#         "bc": "#8C3B0C",   # dark brown
#         "raw": "#52514e"
#         }
STYLES = {"rg": "-", "rc": "--", "bg": "-", "bc": "--", "raw": ":"}
# static expanding-window maps (eq. 9): key -> (label, family whose colour they share); dotted
STATIC = {"rs": ("Static RQS", "rg"), "bs": ("Static Beta", "bg")}


def _lab(k):
    return "Raw pool" if k == "raw" else STATIC[k][0] if k in STATIC else METHODS[k]


def _col(k):
    return COLORS[STATIC[k][1]] if k in STATIC else COLORS[k]


def _sty(k):
    return ":" if k in STATIC else STYLES[k]


def _keys(e, show_static=True):
    """raw, the static maps present in this entry, then the four methods."""
    return ["raw", *(k for k in STATIC if show_static and k in e), *METHODS]
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, 
                     "axes.spines.top": False,
                     "axes.spines.right": False, 
                     "figure.facecolor": "white"})


# ----------------------------------------------------------------- storage
def add_scenario(results, name, raw, rg, rc, bg, bc, group="", rs=None, bs=None):
    """PITs only, each (12, T) with rows in DELTAS order; used by pit_grid.
    rs, bs: optional PITs of the static RQS / Beta maps.
    A (pits, log_t) tuple from the old interface is accepted and log_t ignored."""
    entry = {"group": group, "localized": results.get(name, {}).get("localized", {})}
    arrays = dict(raw=raw, rg=rg, rc=rc, bg=bg, bc=bc)
    arrays.update({k: v for k, v in dict(rs=rs, bs=bs).items() if v is not None})
    for key, arr in arrays.items():
        pits = np.asarray(arr[0] if isinstance(arr, tuple) else arr, float)
        if pits.ndim != 2 or pits.shape[0] != len(DELTAS):
            raise ValueError(f"{name}/{key}: pits must be (12, T), got {pits.shape}")
        entry[key] = {"pits": pits}
    results[name] = entry
    return results


# rows of evaluate_localized's master table used here: label there -> key here
LOCALIZED_ROWS = {
    "PIT KS statistic": "ks",
    "PIT KS p-value": "ks_p",
    "LogS  (unweighted)": "logs",
    "CRPS  (lower better)": "crps",
    "interval score 90% (lower better)": "is90",
    "interval score 95% (lower better)": "is95",
    "CSL  left tail": "csl_l",
    "CSL  right tail": "csl_r",
}
LOCALIZED_NAMES = {"GAS spline": "rg", "CONV spline": "rc", "GAS beta": "bg", "CONV beta": "bc"}
LOCALIZED_NAMES_S = {**LOCALIZED_NAMES, "Static spline": "rs", "Static beta": "bs"}
RAW = "Raw pool"
_GAIN_COL = "delta vs raw pool  (+ = better)"


def add_localized(results, name, loc, names=None):
    """Store, for each row in LOCALIZED_ROWS, the level of the raw pool and of every method
    (from master_levels) and each method's gain over the raw pool with its HAC stars
    (from master), keyed by the result's discount factor.
    `names` maps the labels used in evaluate_localized(methods=...) to rg/rc/bg/bc."""
    if name not in results:
        raise KeyError(f"add_scenario({name!r}, ...) first")
    master, levels = loc.frames["master"], loc.frames["master_levels"]
    if names is None:      # the static maps are stored too when they were evaluated
        names = (LOCALIZED_NAMES_S if all(k in levels.columns for k in ("Static spline", "Static beta"))
                 else LOCALIZED_NAMES)
    for lab in names:
        if lab not in levels.columns:
            raise KeyError(f"method {lab!r} not in the localized result "
                           f"(has {list(levels.columns)}); pass names= to match your labels")
    out = {}
    for label, key in LOCALIZED_ROWS.items():
        if label not in levels.index:
            continue                                   # e.g. no tail region at this cut
        row = {"raw": float(levels.loc[label, RAW]), "level": {}, "gain": {}}
        for lab, m in names.items():
            row["level"][m] = float(levels.loc[label, lab])
            g = master.loc[label, (_GAIN_COL, lab)] if (_GAIN_COL, lab) in master.columns else ""
            row["gain"][m] = str(g).strip()
        out[key] = row
    out["window"] = tuple(loc.window)
    out["n"] = int(loc.n)
    results[name].setdefault("localized", {})[round(float(loc.discount_factor), 6)] = out
    return results


def save_results(results, path):
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_results(path):
    return pickle.load(open(path, "rb")) if Path(path).exists() else {}


def _loc(e, d):
    return e.get("localized", {}).get(round(float(DELTAS[d]), 6))


def _gain_value(sv):
    try:
        return float(str(sv).rstrip("*"))
    except ValueError:
        return np.nan


def _n_stars(sv):
    sv = str(sv).strip()
    return len(sv) - len(sv.rstrip("*"))


def _gain_tex(sv):
    """'+0.0123**' -> '$+0.0123^{**}$'"""
    sv = str(sv).strip()
    if not sv or not np.isfinite(_gain_value(sv)):
        return "--"
    num, stars = sv.rstrip("*"), sv[len(sv.rstrip("*")):]
    return f"${num}^{{{stars}}}$" if stars else f"${num}$"


# ----------------------------------------------------------------- tables
def _num_cell(num, sup="", bold=False):
    """One math-mode cell: '$+0.0979^{***}$' or '$\\mathbf{+0.0979^{***}}$'.
    `num` is the number as text, `sup` the superscript (stars or \\dagger)."""
    body = f"{num}^{{{sup}}}" if sup else num
    return rf"$\mathbf{{{body}}}$" if bold else f"${body}$"


def _split_gain(sv):
    """'+0.0123**' -> ('+0.0123', '**'); unusable input -> (None, '')."""
    sv = str(sv).strip()
    if not sv or not np.isfinite(_gain_value(sv)):
        return None, ""
    num = sv.rstrip("*")
    return num, sv[len(num):]


def _score_methods(results, d=None):
    """Column order for the tables: static, GAS, Con per family when every scenario has the
    static maps in its localized results, else the four methods."""
    order = ("rs", "rg", "rc", "bs", "bg", "bc")
    for e in results.values():
        locs = [_loc(e, d)] if d is not None else list(e.get("localized", {}).values())
        for loc in locs:
            if not loc or not all(m in loc.get("logs", loc.get("ks", {})).get("level", {}) for m in ("rs", "bs")):
                return tuple(METHODS)
    return order


def scorecard_latex(results, d, caption=None, static=True):
    """Main-text table at discount-factor index d. Every row comes from evaluate_localized:
    KS (level, dagger if its p-value < 0.05), then LogS, CRPS, IS90, IS95, CSL_L, CSL_R
    (raw-pool level, then each method's gain with HAC stars). Best method in bold."""
    rows = [("logs", "LogS"), ("crps", "CRPS"), ("is90", r"IS$_{90}$"), ("is95", r"IS$_{95}$"),
            ("csl_l", r"CSL$_{L}$"), ("csl_r", r"CSL$_{R}$")]
    meths = _score_methods(results, d) if static else tuple(METHODS)

    lines = [r"\begin{table}[H]\centering\small",
             r"\begin{tabular}{ll" + "c" * (1 + len(meths)) + r"}\toprule",
             "Scenario & Metric & Raw pool & " + " & ".join(_lab(m) for m in meths) + r" \\ \midrule"]
    windows = set()
    for name, e in results.items():
        loc = _loc(e, d)
        if not loc:
            raise ValueError(f"{name}: no evaluate_localized result at delta={DELTAS[d]}; "
                             f"run sr.add_localized(results, {name!r}, loc) with that delta")
        windows.add((loc["window"], loc["n"]))
        first = True

        if "ks" in loc:
            ks, ksp = loc["ks"], loc.get("ks_p")

            def dag(m):
                if ksp is None:
                    return ""
                p = ksp["raw"] if m == "raw" else ksp["level"][m]
                return r"\dagger" if p < .05 else ""

            win = min(meths, key=lambda m: ks["level"][m])
            cells = [_num_cell(f"{ks['raw']:.3f}", dag("raw"))] + \
                    [_num_cell(f"{ks['level'][m]:.3f}", dag(m), m == win) for m in meths]
            lines.append(f"{name} & KS & " + " & ".join(cells) + r" \\")
            first = False

        for key, label in rows:
            if key not in loc:
                continue
            r = loc[key]
            vals = {m: _gain_value(r["gain"][m]) for m in meths}
            win = max(vals, key=lambda m: vals[m] if np.isfinite(vals[m]) else -np.inf)
            cells = [_num_cell(f"{r['raw']:.4f}")]
            for m in meths:
                num, stars = _split_gain(r["gain"][m])
                cells.append("--" if num is None else _num_cell(num, stars, m == win))
            lines.append(f"{name if first else ''} & {label} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule\end{tabular}"

    win_txt = ""
    if len(windows) == 1:
        (w, n), = windows
        win_txt = rf" All metrics are computed on the evaluation window $[{w[0]},{w[1]})$ ($n={n}$)."
    cap = caption or (rf"Simulation results at LDF discount factor $\delta={DELTAS[d]}$.{win_txt} "
                      r"KS: Kolmogorov--Smirnov distance of the PIT from uniform ($^\dagger$ rejects at 5\%). "
                      r"Raw pool column: level of each score for the uncalibrated pool (LogS, CSL: higher "
                      r"is better; CRPS, IS: lower is better). Method columns: gains over the raw pool, "
                      r"oriented so that positive is better. LogS: unweighted log score; CRPS; IS$_{90}$, "
                      r"IS$_{95}$: 90\% and 95\% interval scores; CSL$_{L}$, CSL$_{R}$: censored likelihood "
                      r"score in the left and right tail regions. Stars: Giacomini--White test of equal "
                      r"predictive ability (HAC $t$-statistic) at 10/5/1\%. Best recalibration method in bold."
                      + (r" Static RQS, Static Beta: expanding-window static maps of each family."
                         if "rs" in meths else ""))
    lines += [rf"\caption{{{cap}}}", r"\label{tab:sim_scorecard}", r"\end{table}"]
    return "\n".join(lines)


def robustness_latex(results, alpha_stars=2, static=True):
    """Appendix table over every discount factor attached with add_localized.
    Each cell: number of discount factors at which KS rejects uniformity at 5%, and in
    brackets the number with a log-score gain over the raw pool significant at 5%
    (at least `alpha_stars` stars from evaluate_localized's HAC test, positive sign)."""
    meths = _score_methods(results) if static else tuple(METHODS)
    lines = [r"\begin{table}[H]\centering\small",
             r"\begin{tabular}{l" + "c" * (1 + len(meths)) + r"}\toprule",
             "Scenario & Raw pool & " + " & ".join(_lab(m) for m in meths) + r" \\ \midrule"]
    counts = set()
    for name, e in results.items():
        locs = list(e.get("localized", {}).values())
        if not locs:
            raise ValueError(f"{name}: no evaluate_localized results attached")
        counts.add(len(locs))
        ks_raw = sum(l["ks_p"]["raw"] < .05 for l in locs if "ks_p" in l)
        cells = [f"{ks_raw}"]
        for m in meths:
            k = sum(l["ks_p"]["level"][m] < .05 for l in locs if "ks_p" in l)
            g = sum(_n_stars(l["logs"]["gain"][m]) >= alpha_stars and
                    _gain_value(l["logs"]["gain"][m]) > 0 for l in locs if "logs" in l)
            cells.append(f"{k} \\,[{g}]")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    n_txt = f"{counts.pop()} " if len(counts) == 1 else ""
    lines += [r"\bottomrule\end{tabular}",
              rf"\caption{{Robustness across the {n_txt}LDF discount factors: number of $\delta$ values at "
              r"which the Kolmogorov--Smirnov test rejects uniformity of the PIT at 5\%. In brackets: "
              r"number of $\delta$ values with a log-score gain over the raw pool significant at 5\% "
              r"(Giacomini--White, HAC). All quantities from the localized evaluation.}",
              r"\label{tab:sim_robustness}", r"\end{table}"]
    return "\n".join(lines)


# ----------------------------------------------------------------- figure
# def pit_grid(results, d, path=None, bins=20):
#     """One row per scenario: raw PIT histogram | best method's PIT histogram |
#     PIT ECDF against the uniform CDF (45-degree line) for the raw pool and all four methods.
#     'Best' = highest unweighted log score from evaluate_localized at this discount factor."""
#     names = list(results)
#     fig, axes = plt.subplots(len(names), 3, figsize=(12, 4 * len(names)),
#                              gridspec_kw=dict(width_ratios=[1, 1, 1]), squeeze=False)
#     grid = np.linspace(0, 1, 401)
#     for i, n in enumerate(names):
#         e = results[n]
#         loc = _loc(e, d)
#         if not loc or "logs" not in loc:
#             raise ValueError(f"{n}: attach evaluate_localized at delta={DELTAS[d]} to pick the best method")
#         # best = max(METHODS, key=lambda m: loc["logs"]["level"][m])
#         # for ax, key, title in [(axes[i, 0], "raw", "Raw pool"),
#         #                        (axes[i, 1], best, f"Best: {METHODS[best]}")]:
#         #     ax.hist(e[key]["pits"][d], bins=bins, range=(0, 1), density=True,
#         #             color=COLORS[key], edgecolor="white", linewidth=0.8)
#         #     ax.axhline(1, color=INK, lw=0.8, ls="--")
#         #     ax.set_xlim(0, 1)
#         #     ax.set_ylabel("Density", color=INK)
#         #     ax.set_title(title, fontsize=12, color=INK, loc="center")
#         # axes[i, 0].set_title(f"Raw pool", fontsize=12, color=INK, loc="center")
        
#         ax = axes[i, 1]
#         for fam, keys in (("spline", ("rg", "rc")), ("Beta", ("bg", "bc"))):
#             k = max(keys, key=lambda m: loc["ks"]["level"][m])
#             ax.hist(e[k]["pits"][d], bins=bins, range=(0, 1), density=True,
#                     histtype="step", lw=1.6, color=COLORS[k], label=METHODS[k])
#         ax.axhline(1, color=INK, lw=0.8, ls="--")
#         ax.set_title("Recalibrated (best of each family)", fontsize=12, color=INK)
#         ax.legend(frameon=False, fontsize=8)
#         axes[i, 0].set_title(f"Raw pool", fontsize=12, color=INK, loc="center")

#         ax = axes[i, 2]
#         n_obs = e["raw"]["pits"].shape[1]
#         band = 1.36 / np.sqrt(n_obs)                       # 5% KS critical value
#         ax.axhspan(-band, band, color="#e4e2dc", lw=0, zorder=0, label="5% KS band")
#         ax.axhline(0, color=INK, lw=0.8)
#         for key in ["raw", *METHODS]:
#             u = np.sort(e[key]["pits"][d])
#             dev = np.searchsorted(u, grid, side="right") / len(u) - grid
#             name = "Raw pool" if key == "raw" else METHODS[key]
#             ax.plot(grid, dev, color=COLORS[key], ls=STYLES[key],
#                     lw=2.0 if key == "raw" else 1.3, label=name)
#         ax.set_xlim(0, 1)
#         lim = max(0.1, 1.15 * max(np.max(np.abs(
#             np.searchsorted(np.sort(e[k]["pits"][d]), grid, side="right")
#             / e[k]["pits"].shape[1] - grid)) for k in ["raw", *METHODS]))
#         ax.set_ylim(-lim, lim)
#         ax.set_ylabel(r"$\hat H(x) - x$", color=INK)
#         ax.set_title("PIT ECDF minus uniform", fontsize=12, color=INK)
#     for ax in axes[-1]:
#         ax.set_xlabel("PIT", color=INK)
#     handles, labels = axes[0, 2].get_legend_handles_labels()
#     fig.legend(handles, labels, ncol=7, frameon=False, fontsize=11, loc="lower center",
#                bbox_to_anchor=(0.5, -0.05), handlelength=2.4)
#     fig.tight_layout(rect=(0, 0.035, 1, 1))
#     if path:
#         fig.savefig(path, bbox_inches="tight")
#     return fig

# Family by colour, update rule by line style.
# COLORS = {"rg": "#2a78d6", "rc": "#2a78d6",    # spline: blue
#           "bg": "#1baf7a", "bc": "#1baf7a",    # Beta: green
#           "raw": "#52514e"}
# STYLES = {"rg": "-", "rc": "--", "bg": "-", "bc": "--", "raw": ":"}
# static expanding-window maps (eq. 9): key -> (label, family whose colour they share); dotted
STATIC = {"rs": ("Static RQS", "rg"), "bs": ("Static Beta", "bg")}


def _lab(k):
    return "Raw pool" if k == "raw" else STATIC[k][0] if k in STATIC else METHODS[k]


def _col(k):
    return COLORS[STATIC[k][1]] if k in STATIC else COLORS[k]


def _sty(k):
    return ":" if k in STATIC else STYLES[k]


def _keys(e, show_static=True):
    """raw, the static maps present in this entry, then the four methods."""
    return ["raw", *(k for k in STATIC if show_static and k in e), *METHODS]
FAMILIES = {"Spline": ("rg", "rc"), "Beta": ("bg", "bc")}


def _deviation(u, grid):
    u = np.sort(np.asarray(u, float))
    return np.searchsorted(u, grid, side="right") / u.size - grid


RECAL_BINS = 10            # fewer, wider bins for the side-by-side recalibrated bars


def _best_by_family(e, d, name="scenario"):
    """Best spline and best Beta method by unweighted log score at delta index d."""
    loc = _loc(e, d)
    if not loc or "logs" not in loc:
        raise ValueError(f"{name}: attach evaluate_localized at delta={DELTAS[d]} "
                         f"to select the best method by log score")
    logs = loc["logs"]
    return {fam: max(keys, key=lambda m: logs["level"][m]) for fam, keys in FAMILIES.items()}


def hist_ylim(results, d, bins=20, recal_bins=RECAL_BINS, pad=1.08):
    """One y-limit covering every PIT histogram (raw and best recalibrated) of every
    scenario at delta index d, so all histogram panels share the same scale."""
    top = 1.0
    for name, e in results.items():
        dens, _ = np.histogram(e["raw"]["pits"][d], bins=bins, range=(0, 1), density=True)
        top = max(top, dens.max())
        for m in _best_by_family(e, d, name).values():
            dens, _ = np.histogram(e[m]["pits"][d], bins=recal_bins, range=(0, 1), density=True)
            top = max(top, dens.max())
    return (0, 2)#pad * top)


def _draw_raw_hist(ax, e, d, bins, ylim):
    ax.hist(e["raw"]["pits"][d], bins=bins, range=(0, 1), density=True,
            color=COLORS["raw"], edgecolor="white", linewidth=0.6)
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _draw_recal_hist(ax, e, d, best, nb, ylim, legend=True):
    edges = np.linspace(0, 1, nb + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    k = len(best)
    bw = (edges[1] - edges[0]) * 0.84 / k              # width of each bar (k bars per bin)
    for j, m in enumerate(best.values()):
        dens, _ = np.histogram(e[m]["pits"][d], bins=edges, density=True)
        ax.bar(centres + (j - (k - 1) / 2) * bw, dens, width=bw,
               color=_col(m), alpha=0.7, edgecolor="white", linewidth=0.5,
               hatch="//" if m in STATIC else None, label=_lab(m))
    ax.axhline(1, color=INK, lw=0.9, ls="--")
    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, 1.02), ncol=1)


def _draw_deviation(ax, e, d, grid, lim=None, show_static=True):
    keys = _keys(e, show_static)
    pits = {k: np.asarray(e[k]["pits"][d], float) for k in keys}
    band = 1.36 / np.sqrt(pits["raw"].size)          # 5% KS critical value
    ax.axhspan(-band, band, color="#e4e2dc", lw=0, zorder=0, label="5% KS band")
    ax.axhline(0, color=INK, lw=0.8, linestyle="--")
    devs = {k: _deviation(pits[k], grid) for k in keys}
    for k in keys:
        ax.plot(grid, devs[k], color=_col(k), ls=_sty(k),
                lw=1 if k == "raw" else 1.0 if k in STATIC else 0.5, label=_lab(k))
    if lim is None:
        lim = 1.15 * max(band, max(np.max(np.abs(v)) for v in devs.values()))
    ax.set_xlim(0, 1)
    ax.set_ylim(-lim, lim)
    return lim


def _scenario_axes(n, ncols, panel):
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * 3, panel[1] * nrows),
                             squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes


def _label_grid(axes, n, ncols, ylabel):
    for i, ax in enumerate(axes.flat[:n]):
        row, col = divmod(i, axes.shape[1])
        if col == 0:
            ax.set_ylabel(ylabel, color=INK)
        if i + ncols >= n:                     # bottom panel of its column
            ax.set_xlabel("PIT", color=INK)
        else:
            ax.tick_params(labelbottom=False)


def _title(name):
    return name.replace("_", " ")


def raw_hist_all(results, d, path=None, bins=20, ncols=4, panel=(4, 3), ylim="shared"):
    """Raw-pool PIT histogram of every scenario in one figure.
    ylim='shared' uses hist_ylim (same scale as recal_hist_all); None lets each panel scale."""
    names = list(results)
    if isinstance(ylim, str):
        ylim = hist_ylim(results, d, bins=bins)
    fig, axes = _scenario_axes(len(names), ncols, panel)
    for ax, name in zip(axes.flat, names):
        _draw_raw_hist(ax, results[name], d, bins, ylim)
        ax.set_title(_title(name), fontsize=12, color=INK, pad=14)
        ax.tick_params(axis="both", colors=INK)
    _label_grid(axes, len(names), axes.shape[1], "Density")
    #fig.suptitle(rf"Raw pool PIT histograms ($\delta={DELTAS[d]}$)", fontsize=12, color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=600)
    #return fig


def recal_hist_all(results, d, path=None, nb=RECAL_BINS, ncols=4, panel=(4, 3),
                   ylim="shared", bins=20):
    """Best spline vs best Beta PIT histograms (by log score) for every scenario in one
    figure, same format as the middle column of pit_grid. `bins` is only used to compute
    the shared y-limit with the raw histograms."""
    names = list(results)
    if isinstance(ylim, str):
        ylim = hist_ylim(results, d, bins=bins, recal_bins=nb)
    fig, axes = _scenario_axes(len(names), ncols, panel)
    for ax, name in zip(axes.flat, names):
        _draw_recal_hist(ax, results[name], d, _best_by_family(results[name], d, name), nb, ylim)
        ax.set_title(_title(name), fontsize=12, color=INK, pad=14)
        ax.tick_params(axis="both", colors=INK)
    _label_grid(axes, len(names), axes.shape[1], "Density")
    #fig.suptitle(rf"Recalibrated PIT histograms, best RQS vs best Beta ($\delta={DELTAS[d]}$)", fontsize=12, color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=600)
    #return fig


def ecdf_dev_all(results, d, path=None, ncols=4, panel=(4, 3), share_y=False, show_static=True):
    """PIT ECDF minus uniform, H(x) - x, with the 5% KS band, for every scenario in one figure.
    Static expanding-window maps ('rs', 'bs') are drawn dotted in their family colour when present.
    share_y=True puts all panels on the largest deviation's scale."""
    names = list(results)
    grid = np.linspace(0, 1, 401)
    keys = lambda e: ["raw", *(k for k in STATIC if show_static and k in e), *METHODS]

    lim = None
    if share_y:
        lim = 1.15 * max(max(1.36 / np.sqrt(e["raw"]["pits"].shape[1]),
                             max(np.max(np.abs(_deviation(e[k]["pits"][d], grid))) for k in keys(e)))
                         for e in results.values())

    def style(k):
        if k == "raw":
            return "Raw pool", "#7f7f7f", "-", 1.4
        if k in STATIC:
            lab, fam = STATIC[k]
            return lab, COLORS[fam], ":", 1.5
        return METHODS[k], COLORS[k], STYLES[k], 1.1

    fig, axes = _scenario_axes(len(names), ncols, panel)
    for ax, name in zip(axes.flat, names):
        e = results[name]
        n = e["raw"]["pits"].shape[1]
        band = 1.36 / np.sqrt(n)
        ax.axhspan(-band, band, color="#e8eef7", lw=0, zorder=0)
        ax.axhline(0, color=INK, lw=0.7, zorder=1)
        for k in keys(e):
            lab, col, ls, lw = style(k)
            ax.plot(grid, _deviation(e[k]["pits"][d], grid), color=col, ls=ls, lw=lw,
                    label=lab, zorder=3 if k != "raw" else 2)
        ax.set_xlim(0, 1)
        if lim is not None:
            ax.set_ylim(-lim, lim)
        ax.set_title(_title(name), fontsize=12, color=INK, pad=14)
        ax.tick_params(axis="both", colors=INK)
    _label_grid(axes, len(names), axes.shape[1], r"$\hat H(x) - x$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#e8eef7", lw=0)); labels.append("5% KS band")
    leg = fig.legend(handles, labels, ncol=min(len(labels), 8), frameon=False, fontsize=10,
                     loc="lower center", bbox_to_anchor=(0.5, -0.04), handlelength=2.6)
    for line in leg.get_lines():
        line.set_linewidth(1.6)                # panel lines are thin; keep the key readable
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=600)


# def _draw_method_hist(ax, e, d, m, nb, ylim, legend=True):
#     """PIT histogram of one recalibration method, same style as _draw_recal_hist."""
#     edges = np.linspace(0, 1, nb + 1)
#     dens, _ = np.histogram(e[m]["pits"][d], bins=edges, density=True)
#     ax.bar(0.5 * (edges[:-1] + edges[1:]), dens, width=(edges[1] - edges[0]) * 0.84,
#            color=COLORS[m], alpha=0.7, edgecolor="white", linewidth=0.5, label=METHODS[m])
#     ax.axhline(1, color=INK, lw=0.9, ls="--")
#     ax.set_xlim(0, 1)
#     if ylim is not None:
#         ax.set_ylim(*ylim)
#     if legend:
#         ax.legend(frameon=False, fontsize=8, loc="upper center",
#                   bbox_to_anchor=(0.5, 1.02), ncol=1)


def pit_grid(results, d, path=None, bins=20, row_height=2.4, ylim="shared", show_static=True):
    """One row per scenario, four panels:
       (1) PIT histogram of the raw pool;
       (2) PIT histograms of Static Beta, Beta-GAS and Beta-Con, side by side;
       (3) the same for RQS;
       (4) deviation of the PIT ECDF from the uniform CDF, H(x) - x, for the raw pool, the
           static maps and the four methods, with the 5% KS band.
    The static maps need their PITs in results[name]['rs'] / ['bs'] (add_scenario(..., rs=, bs=)).
    ylim='shared' gives every histogram panel the same y-scale; None lets each panel scale."""
    names = list(results)
    panels = {"Beta": ("bs", {"GAS": "bg", "Con": "bc"}),
              "RQS":  ("rs", {"GAS": "rg", "Con": "rc"})}
    missing = [n for n in names if show_static and not all(k in results[n] for k in ("rs", "bs"))]
    if missing:
        print(f"pit_grid: no static PITs for {missing}; pass rs=, bs= to add_scenario to show them")

    if isinstance(ylim, str):                      # shared scale over raw, static and all methods
        top = 1.0
        for e in results.values():
            dens, _ = np.histogram(e["raw"]["pits"][d], bins=bins, range=(0, 1), density=True)
            top = max(top, dens.max())
            keys = ["rc", "rg", "bc", "bg"] + ([k for k in ("rs", "bs") if k in e] if show_static else [])
            for m in keys:
                dens, _ = np.histogram(e[m]["pits"][d], bins=RECAL_BINS, range=(0, 1), density=True)
                top = max(top, dens.max())
        ylim = (0, 1.08 * top)
    fig, axes = plt.subplots(len(names), 4, figsize=(15, 4 * len(names)),
                             gridspec_kw=dict(width_ratios=[1, 1.15, 1.15, 1.1]),
                             squeeze=False)
    grid = np.linspace(0, 1, 401)

    for i, name in enumerate(names):
        e = results[name]

        ax = axes[i, 0]
        _draw_raw_hist(ax, e, d, bins, ylim)
        ax.set_ylabel("Density", color=INK)
        if i == 0:
            ax.set_title("Raw pool", fontsize=12, color=INK)

        for j, (fam, (st, pair)) in enumerate(panels.items(), start=1):
            ax = axes[i, j]
            if show_static and st in e:
                pair = {"Static": st, **pair}        # static (hatched), GAS, Con
            _draw_recal_hist(ax, e, d, pair, RECAL_BINS, ylim)
            if i == 0:
                ax.set_title(fam, fontsize=12, color=INK)

        ax = axes[i, 3]
        _draw_deviation(ax, e, d, grid, show_static=show_static)
        ax.set_ylabel(r"$\hat H(x) - x$", color=INK)
        if i == 0:
            ax.set_title("PIT ECDF minus uniform", fontsize=12, color=INK)

    for ax in axes[-1]:
        ax.set_xlabel("PIT", color=INK)
    for ax in axes[:-1].ravel():
        ax.tick_params(labelbottom=False)

    handles, labels = axes[0, 3].get_legend_handles_labels()
    leg = fig.legend(handles, labels, ncol=min(len(labels), 8), frameon=False, fontsize=10,
                     loc="lower center", bbox_to_anchor=(0.5, -0.05), handlelength=2.6)
    for line in leg.get_lines():
        line.set_linewidth(1.6)
    for ax in axes.ravel():
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(plt.rcParams["grid.color"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=600)
    #return fig



TEST_LABELS = ["KS", "AD", r"$\chi^2$", "DH", "LB", "Berkowitz"]
TEST_ROWS = ["Raw pool", *METHODS.values()]
TEST_ROWS_S = ["Raw pool", "Static RQS", "RQS-GAS", "RQS-Con", "Static Beta", "Beta-GAS", "Beta-Con"]

def _draw_tests_heatmap(ax, counts, n_max=len(DELTAS), cmap="Blues", ylabels=True, fontsize=8):
    """Heatmap of non-rejections (of n_max discount factors) on one axis, in the style of
    sim_tests_heatmap. counts: 5 rows (raw pool, RQS-GAS, RQS-Con, Beta-GAS, Beta-Con), or 7 with the static
    maps (raw, Static RQS, RQS-GAS, RQS-Con, Static Beta, Beta-GAS, Beta-Con)
    x 6 tests (KS, AD, chi2, DH, LB, Berkowitz). White lines split uniformity tests from
    independence tests, and the raw pool from the recalibrated methods."""
    M = np.asarray(counts)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=n_max, aspect="auto")
    for i, j in np.ndindex(M.shape):
        ax.text(j, i, M[i, j], ha="center", va="center", fontsize=fontsize,
                color="white" if M[i, j] >= 2 * n_max / 3 else "#2b2b2b")
    ax.axvline(3.5, color="white", lw=1.5, zorder=3)          # uniformity | independence
    ax.axhline(0.5, color="white", lw=1.5, zorder=3)          # raw pool | recalibrated
    rows = TEST_ROWS_S if M.shape[0] == len(TEST_ROWS_S) else TEST_ROWS
    if M.shape[0] == len(TEST_ROWS_S):
        ax.axhline(3.5, color="white", lw=1.5, zorder=3)      # RQS block | Beta block
    ax.set_xticks(range(len(TEST_LABELS)), TEST_LABELS, fontsize=fontsize,color=INK)
    ax.set_yticks(range(len(rows)), rows if ylabels else [], fontsize=fontsize,color=INK)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im
 
 
def pit_panel(results, name, d, tests, path=None, nb=RECAL_BINS, ylim="shared",
              figsize=(8, 6), cmap="Blues", show_static=True):
    """2x2 figure for ONE scenario at delta index d:
       top left     PIT histograms of both RQS methods (GAS and Con) side by side
       top right    PIT histograms of both Beta methods (GAS and Con) side by side
       bottom left  PIT ECDF minus uniform, H(x) - x, raw pool and all four methods, 5% KS band
       bottom right heatmap of tests passed (out of the 12 discount factors)
    tests: the 5 x 6 counts for this scenario, e.g. TVD_RESULTS["Time_varying_dispersion"]
    (rows raw pool, RQS-GAS, RQS-Con, Beta-GAS, Beta-Con; columns KS, AD, chi2, DH, LB, Berkowitz).
    ylim='shared' puts both histogram panels on one y-scale (largest density among the four
    methods); pass a (lo, hi) tuple to set it, or None to let each panel scale on its own."""
    e = results[name]
    if isinstance(ylim, str):
        top = 1.0
        for m in METHODS:
            dens, _ = np.histogram(e[m]["pits"][d], bins=nb, range=(0, 1), density=True)
            top = max(top, dens.max())
        ylim = (0, 2)
 
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    grid = np.linspace(0, 1, 401)
 
    for ax, (fam, keys) in zip(axes[0], FAMILIES.items()):
        pair = dict(zip(("GAS", "Con"), keys))
        st = "rs" if fam == "Spline" else "bs"
        if show_static and st in e:
            pair = {"Static": st, **pair}
        _draw_recal_hist(ax, e, d, pair, nb, ylim)
        ax.set_title("RQS" if fam == "Spline" else fam, fontsize=12, color=INK, pad=14)
        ax.set_xlabel("PIT", color=INK)
        ax.tick_params(color=INK, labelcolor=INK)
    axes[0, 0].set_ylabel("Density", color=INK)
    
 
    ax = axes[1, 0]
    _draw_deviation(ax, e, d, grid, show_static=show_static)
    ax.set_title("PIT ECDF minus uniform", fontsize=12, color=INK)
    ax.set_xlabel("PIT", color=INK)
    ax.set_ylabel(r"$\hat H(x) - x$", color=INK)
    ax.tick_params(color=INK, labelcolor=INK)
 
    ax = axes[1, 1]
    im = _draw_tests_heatmap(ax, tests, cmap=cmap, fontsize=9)
    ax.set_title("Tests passed", fontsize=12, color=INK)
    cb = fig.colorbar(im, ax=ax, location="right", shrink=0.9, pad=0.02)
    cb.set_label(f"Non-rejections (of {len(DELTAS)})", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
 
    handles, labels = axes[1, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, ncol=min(len(labels), 4), frameon=False, fontsize=10,
                     loc="lower center", bbox_to_anchor=(0.5, -0.06), handlelength=2.6)
    for line in leg.get_lines():
        line.set_linewidth(1.6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=600)