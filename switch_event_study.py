import numpy as np
import matplotlib.pyplot as plt
import sim_report as sr
import replicate_sim as rp

# style for the static maps: family colour, dotted
EXTRA = {"rs": ("Static RQS", sr.COLORS["rg"], ":"),
         "bs": ("Static Beta", sr.COLORS["bg"], ":")}
_label = lambda m: EXTRA[m][0] if m in EXTRA else sr.METHODS[m]
_color = lambda m: EXTRA[m][1] if m in EXTRA else sr.COLORS[m]
_style = lambda m: EXTRA[m][2] if m in EXTRA else sr.STYLES[m]


def runs_from_reps(reps, name, delta):
    """One run per replication of scenario `name`: regime path plus each method's per-period
    gain over the raw pool at discount factor `delta`, including the static maps (rs, bs)."""
    i = rp._didx(delta)
    runs = []
    for r in reps:
        if r["name"] != name or r.get("state") is None:
            continue
        run = {"state": np.asarray(r["state"])}
        for m in ("rg", "rc", "bg", "bc"):
            run[m] = np.asarray(r["gain"][m][i], float)
        run["rs"] = np.asarray(r["static_gain"]["rg"][i], float)
        run["bs"] = np.asarray(r["static_gain"]["bg"][i], float)
        runs.append(run)
    return runs


def switch_times(state, start, pre, post, min_gap=0):
    """Indices t where S_t != S_{t-1}, keeping only events whose [t-pre, t+post] window lies
    inside [start, T). min_gap drops a switch that follows another within min_gap periods."""
    s = np.asarray(state).astype(int)
    tt = np.flatnonzero(np.diff(s) != 0) + 1
    if min_gap:
        keep, last = [], -np.inf
        for t in tt:
            if t - last >= min_gap:
                keep.append(t)
            last = t
        tt = np.asarray(keep, int)
    return tt[(tt - pre >= start) & (tt + post < s.size)]


def event_study(runs, methods=("bs", "bg", "bc", "rs", "rg", "rc"), pre=5, post=30, eval_start=500,
                by_direction=True, min_gap=0, level=0.95):
    """Average per-period log-score gain around regime switches.
    runs: list of dicts {"state": (T,), method: (T,) gain over raw}; static maps under 'rs', 'bs'.
    Returns {direction: {method: (mean, lo, hi, n_events)}} on lags -pre..post."""
    lags = np.arange(-pre, post + 1)
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[level]
    dirs = ({r"$S_t = 0$": 0, r"$S_t \neq 0$": 1} if by_direction else {"all switches": None})
    out = {}
    for dname, target in dirs.items():
        out[dname] = {}
        for m in methods:
            rows = []
            for run in runs:
                st = np.asarray(run["state"]).astype(int)
                g = np.asarray(run[m], float)
                for t in switch_times(st, eval_start, pre, post, min_gap):
                    if target is None or st[t] == target:
                        rows.append(g[t + lags])
            R = np.asarray(rows)
            mean = R.mean(axis=0)
            se = R.std(axis=0, ddof=1) / np.sqrt(len(R))
            out[dname][m] = (mean, mean - z * se, mean + z * se, len(R))
    return lags, out


def plot_event_study(runs, methods=("bs", "bg", "bc", "rs", "rg", "rc"), pre=5, post=30,
                     eval_start=500, by_direction=True, min_gap=0, level=0.95, path=None,
                     figsize=None):
    """Event-study figure: per-period log-score gain over the raw pool around regime switches,
    with a confidence band across switches; static maps dotted in their family colour."""
    lags, res = event_study(runs, methods, pre, post, eval_start, by_direction, min_gap, level)
    n = len(res)
    fig, axes = plt.subplots(1, n, figsize=figsize or (6.5 * n, 4.2), sharey=True, squeeze=False)
    for ax, (dname, d) in zip(axes[0], res.items()):
        ax.axvspan(lags[0], 0, color="#f3f2ee", lw=0, zorder=0)
        ax.axhline(0, color=sr.INK, lw=0.8, ls=":")
        ax.axvline(0, color=sr.INK, lw=0.8, ls="--")
        for m in methods:
            mean, lo, hi, k = d[m]
            if m not in EXTRA:                       # bands for the adaptive methods only
                ax.fill_between(lags, lo, hi, color=_color(m), alpha=0.15, lw=0)
            ax.plot(lags, mean, color=_color(m), ls=_style(m), lw=2.0 if m in EXTRA else 1.8,
                    label=_label(m))
        k = d[methods[0]][3]
        ax.set_title(f"{dname}", fontsize=11, color=sr.INK)
        ax.set_xlabel("Periods since regime switch", color=sr.INK)
        ax.set_xlim(lags[0], lags[-1])
    axes[0, 0].set_ylabel("Mean log-score gain", color=sr.INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=1, frameon=False, fontsize=10,
               loc="upper right", bbox_to_anchor=(1, 1))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")