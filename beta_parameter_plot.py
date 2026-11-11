import numpy as np
import matplotlib.pyplot as plt

A_COL, B_COL, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#898781"
GAS_COL, CON_COL = "#898781", "#2a78d6"

REGIME_SHADE = "#e8eef7"


def _spans(mask):
    m = np.asarray(mask, bool)
    cuts = np.flatnonzero(np.diff(m.astype(int))) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, m.size]
    return [(s, e) for s, e in zip(starts, ends) if m[s]]


def _rolling_mean(x, w, center=True):
    """Moving average over w periods, ignoring NaNs.

    center=True  window [t - w//2, t + w - w//2): no delay, so a step in the series
                 stays where it happened. Right for a DISPLAY smoother.
    center=False trailing window [t - w + 1, t]: causal, but shifts every step
                 about w/2 periods later, which makes a sharp switch look like a lag."""
    x = np.asarray(x, float)
    if w is None or w <= 1:
        return x
    ok = np.isfinite(x)
    c = np.cumsum(np.insert(np.where(ok, x, 0.0), 0, 0.0))
    k = np.cumsum(np.insert(ok.astype(float), 0, 0.0))
    i = np.arange(len(x))
    lo = np.maximum(i - w // 2, 0) if center else np.maximum(i - w + 1, 0)
    hi = np.minimum(lo + w, len(x)) if center else i + 1
    cnt = k[hi] - k[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _shade(ax, t, state, shade_regime, t0, t1):
    """Shade every period whose regime equals shade_regime. Spans run from the first
    period of a spell to the first period of the next one, so the last period of the
    window is shaded too."""
    if state is None:
        return
    st = np.asarray(state)
    if st.shape[0] < t1:
        raise ValueError(f"state has {st.shape[0]} periods but the plot needs {t1}")
    for s0, s1 in _spans(st[t0:t1] == shade_regime):
        ax.axvspan(t0 + s0, t0 + s1, color=REGIME_SHADE, lw=0, zorder=0)


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.grid(axis="x", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))


def _ylim(ax, vals, q_lo, q_hi, ref=1.0):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [q_lo, q_hi])
    lo, hi = min(lo, ref), max(hi, ref)
    pad = 0.08 * (hi - lo) if hi > lo else 0.05 * max(abs(ref), 1.0)
    ax.set_ylim(max(lo - pad-0.1, 0), hi + pad+0.2)


# --------------------------------------------------------------------------- #
#  Effective Beta parameters of the DEPLOYED map
# --------------------------------------------------------------------------- #
def _quad_weights(u):
    """Quadrature weights on [0, 1] for the stored nodes u (rep['u_grid']).

    The recalibrators store tau on recal_dma_new's _UN grid. Use its own weights _UW when
    they can be imported and match; otherwise accept u only if it IS the Gauss-Legendre
    grid of that size (checked, not assumed); otherwise fall back to the trapezoid rule
    with the end points tau(0) = 0, tau(1) = 1 added."""
    u = np.asarray(u, float)
    try:
        from recal_dma_new import _UN, _UW
        if len(_UN) == len(u) and np.allclose(_UN, u):
            w = np.asarray(_UW, float)
            return u, w / w.sum(), None          # normalise: sum of weights = length 1
    except ImportError:
        pass
    x, w = np.polynomial.legendre.leggauss(len(u))
    if np.allclose(0.5 * (x + 1.0), u, atol=1e-10):
        return u, 0.5 * w, None
    return u, None, "trapezoid"


def map_moments(rep):
    """Mean and variance of the deployed map tau_t, read as a CDF on (0, 1).

        m_t = E[V] = 1 - int tau_t(u) du,   E[V^2] = 1 - 2 int u tau_t(u) du

    Identity map: m = 1/2, var = 1/12. Returns (m, var), each (T,)."""
    tau = np.asarray(rep["tau_grid"], float)
    u, w, rule = _quad_weights(rep["u_grid"])
    if rule == "trapezoid":
        uu = np.r_[0.0, u, 1.0]
        tt = np.concatenate([np.zeros((len(tau), 1)), tau, np.ones((len(tau), 1))], axis=1)
        trap = getattr(np, "trapezoid", None) or np.trapz          # numpy < 2.0 has only trapz
        m = 1.0 - trap(tt, uu, axis=1)
        s = 1.0 - 2.0 * trap(tt * uu, uu, axis=1)
    else:
        m = 1.0 - tau @ w
        s = 1.0 - 2.0 * tau @ (u * w)
    return m, np.maximum(s - m ** 2, 1e-12)


def effective_ab(rep):
    """Effective (a_t, b_t) of the map actually applied at each t: the Beta with the same
    mean and variance as tau_t (moment matching).

    a_path / b_path (GAS) and a_gas / b_gas (Con) are softplus(weighted mean of the bank
    logits). They are NOT the parameters of the deployed map, which is the mixture
    sum_k w_k I_u(a_k, b_k) and, for Con, its composition with the NN layer. tau_grid is
    the deployed map for both methods, so GAS and the full Con composition are compared
    like for like. The identity map gives a = b = 1 exactly; a single Beta map is
    recovered exactly; a mixture or composition gets its closest Beta by two moments.

        k = m(1-m)/var - 1,   a = m k,   b = (1-m) k

    var < m(1-m) for any law on (0,1); k is floored at 1e-6 so quadrature error near a
    two-point map cannot produce a negative parameter."""
    m, v = map_moments(rep)
    k = np.maximum(m * (1.0 - m) / v - 1.0, 1e-6)
    return m * k, (1.0 - m) * k


def map_sd(rep):
    """SD of the deployed map tau_t, read as a distribution on (0, 1) (identity = 1/sqrt(12) = 0.289).
    Below 0.289 (e.g. a, b > 1) tau is steep in the middle: it spreads central PITs apart and
    NARROWS the predictive -- the correction for an over-dispersed pool (hump-shaped PIT).
    Above 0.289 (a, b < 1) it WIDENS the predictive -- the correction for an under-dispersed
    pool (U-shaped PIT)."""
    return np.sqrt(map_moments(rep)[1])


def map_mean(rep):
    """Mean of the deployed map tau_t, read as a distribution on (0, 1) (identity = 0.5).
    Below 0.5, tau(u) > u: the map shifts the predictive DOWN -- the correction for a pool
    that is too high (PITs skewed towards 0). Above 0.5 it shifts the predictive UP."""
    return map_moments(rep)[0]


# --------------------------------------------------------------------------- #
#  Original figure: layer parameters (kept for the appendix)
# --------------------------------------------------------------------------- #
def plot_beta_params(bg_rep, bc_rep, eval_start, path=None, q_lo=0.5, q_hi=99.5,
                     state=None, shade_regime=False, t_range=None,
                     shade_label=r"$S_t=0$ (members dispersed)"):
    """Three stacked panels on a shared time axis, each with its own y-range:
    GAS-Beta a, b | Convolution: GAS layer a, b | Convolution: NN layer a, b.

    NOTE: GAS panels show softplus(mean logit), not the deployed mixture map.
    Use plot_beta_effective for the main-text comparison.
    """
    T = len(bg_rep["a_path"])
    t0, t1 = (eval_start, T) if t_range is None else (max(eval_start, t_range[0]), min(T, t_range[1]))
    t = np.arange(T)[t0:t1]
    panels = [
        ("GAS-Beta", bg_rep["a_path"], bg_rep["b_path"]),
        ("Convolution-Beta: GAS layer", bc_rep["a_gas"], bc_rep["b_gas"]),
        ("Convolution-Beta: NN layer", bc_rep["a_nn"], bc_rep["b_nn"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 0.7]))
    for ax, (title, a, b) in zip(axes, panels):
        a, b = np.asarray(a)[t0:t1], np.asarray(b)[t0:t1]
        _shade(ax, t, state, shade_regime, t0, t1)
        ax.axhline(1, color=MUTED, lw=1, ls="--", zorder=1)
        ax.plot(t, a, color=A_COL, lw=0.9, label="$a_t$", zorder=3)
        ax.plot(t, b, color=B_COL, lw=0.9, ls=(0, (4, 1.5)), label="$b_t$", zorder=2)
        _ylim(ax, np.concatenate([a, b]), q_lo, q_hi)
        _style(ax, title)
    axes[1].set_ylabel("Beta parameter", fontsize=11, color=INK)
    axes[-1].set_xlabel("Time", fontsize=11, color=INK)
    axes[-1].set_xlim(t[0], t[-1])
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, ls="--", lw=1))
    labels.append("Identity ($a=b=1$)")
    if state is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=REGIME_SHADE, lw=0))
        labels.append(shade_label)
    fig.legend(handles, labels, ncol=len(labels), frameon=False, fontsize=10,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    if path:
        fig.savefig(path, bbox_inches="tight")
    #return fig


# --------------------------------------------------------------------------- #
#  New figure: effective parameters of the deployed maps, GAS vs Con
# --------------------------------------------------------------------------- #
def plot_beta_effective(bg_rep, bc_rep, eval_start, path=None, q_lo=None, q_hi=None,
                        state=None, shade_regime=False, t_range=None, smooth=5, center=True,
                        panels=("a", "b", "sd"),y_lim1=None,y_lim2=None,
                        shade_label=r"$S_t=0$", figsize=None, show_nn=False):
    """Effective parameters of the map ACTUALLY applied, read from tau_grid, GAS-Beta vs
    Convolution-Beta (full composition). One line per method; the y-axis label names the quantity.

    smooth   centred moving average over `smooth` periods (no delay, so a step in the Con map
             stays at the switch); None or 1 plots the raw per-period values instead.
    q_lo, q_hi  percentiles that set each panel's y-range. Default: the full range (0, 100) for
             raw values, so no spike is cut off; (0.5, 99.5) when smoothing.
    show_nn  also draw the composition's network layer (a_nn, b_nn) in the a and b panels.
             It is the inner layer only, not the applied map.
    state    (T,) regime; periods with state == shade_regime are shaded."""
    spec = {"a": (r"$a_t$", effective_ab, 0, 1.0),
            "b": (r"$b_t$", effective_ab, 1, 1.0),
            "sd": ("Map SD", map_sd, None, 1.0 / np.sqrt(12.0)),
            "mean": ("Map mean", map_mean, None, 0.5)}
    unknown = set(panels) - set(spec)
    if unknown:
        raise ValueError(f"unknown panels {sorted(unknown)}; choose from {sorted(spec)}")
    use_smooth = bool(smooth and smooth > 1)
    if q_lo is None:
        q_lo = 0.5 if use_smooth else 0.0
    if q_hi is None:
        q_hi = 99.5 if use_smooth else 100.0

    cache = {}
    def series(rep, key):
        _, fn, j, _ = spec[key]
        tag = (id(rep), fn)
        if tag not in cache:
            cache[tag] = fn(rep)
        out = cache[tag]
        return np.asarray(out[j] if j is not None else out, float)

    T = len(np.asarray(bg_rep["tau_grid"]))
    if len(np.asarray(bc_rep["tau_grid"])) != T:
        raise ValueError("GAS and Con reports have different lengths")
    t0, t1 = (eval_start, T) if t_range is None else (max(eval_start, t_range[0]), min(T, t_range[1]))
    if t1 <= t0:
        raise ValueError(f"empty window [{t0}, {t1})")
    t = np.arange(t0, t1)

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=figsize or (11, 2.6 * n + 0.6), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    for ax, key in zip(axes, panels):
        ylab, _, _, ref = spec[key]
        g_full, c_full = series(bg_rep, key), series(bc_rep, key)
        if use_smooth:      # smoothed on the full series so the window edge at t0 is not truncated
            g_full = _rolling_mean(g_full, smooth, center)
            c_full = _rolling_mean(c_full, smooth, center)
        g, c = g_full[t0:t1], c_full[t0:t1]
        _shade(ax, t, state, shade_regime, t0, t1)
        ax.axhline(ref, color=MUTED, lw=1, ls="--", zorder=1)
        ax.plot(t, g, color=GAS_COL, lw=1, label="Beta-GAS", zorder=4)
        ax.plot(t, c, color=CON_COL, lw=1, label="Beta-Con", zorder=5)
        vals = [g, c]
        if show_nn and key in ("a", "b"):
            if "a_nn" not in bc_rep:
                raise KeyError("bc_rep has no a_nn/b_nn: pass the convolution report as bc_rep")
            nn_full = np.asarray(bc_rep["a_nn" if key == "a" else "b_nn"], float)
            if use_smooth:
                nn_full = _rolling_mean(nn_full, smooth, center)
            nn = nn_full[t0:t1]
            ax.plot(t, nn, color=CON_COL, lw=1, ls=":", label="Beta-Con: network layer", zorder=6)
            vals.append(nn)
        ax.set_ylim(y_lim1, y_lim2)
        #_ylim(ax, np.concatenate(vals), q_lo, q_hi, ref=ref)
        _style(ax, "")
        ax.set_ylabel(ylab, fontsize=12, color=INK)
    axes[-1].set_xlabel("Time", fontsize=11, color=INK)
    axes[-1].set_xlim(t0, t1 - 1)
    # legend from every panel, so the network-layer entry appears even if the first panel is "sd"
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); handles.append(h); labels.append(l)
    handles.append(plt.Line2D([], [], color=MUTED, ls="--", lw=1))
    labels.append("Identity map")
    if state is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=REGIME_SHADE, lw=0))
        labels.append(shade_label)
    fig.legend(handles, labels, ncol=len(labels), frameon=False, fontsize=10,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig