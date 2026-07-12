import json, math
import numpy as np
from scipy.optimize import curve_fit

def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def survival_curve(rows, condition):
    ks = sorted({r["k"] for r in rows if r["condition"] == condition})
    S = []
    for k in ks:
        sub = [r["correct"] for r in rows if r["condition"] == condition and r["k"] == k]
        S.append(sum(sub)/len(sub))
    return ks, S

def empirical_chance(rows):
    # Guess rate from the 'nofact' baseline arm (query asked with NO planted info).
    # Empirical floor per (model, ftype); replaces hand-coded chance_level.
    nf = [r for r in rows if r["condition"] == "nofact"]
    out = {"overall": (sum(r["correct"] for r in nf) / len(nf)) if nf else None, "by_ftype": {}}
    for ft in {r["ftype"] for r in nf}:
        sub = [r["correct"] for r in nf if r["ftype"] == ft]
        out["by_ftype"][ft] = sum(sub) / len(sub)
    return out

def _model(k, tau, f, S0):
    return f + (S0 - f) * np.exp(-np.asarray(k, float)/tau)

def fit_tau(ks, S, chance):
    ks = np.asarray(ks, float); S = np.asarray(S, float)
    p0 = [max(ks.max()/2, 1.0), min(S.min(), chance), S[0]]
    try:
        popt, _ = curve_fit(_model, ks, S, p0=p0, maxfev=20000,
                            bounds=([1e-3, -0.2, 0.0], [1e6, 1.0, 1.0001]))
    except Exception:
        popt = p0
    resid = S - _model(ks, *popt)
    ss_res = float((resid**2).sum())
    ss_tot = float(((S - S.mean())**2).sum()) or 1e-9
    return {"tau": float(popt[0]), "f": float(popt[1]), "S0": float(popt[2]),
            "r2": 1 - ss_res/ss_tot}

def is_below_floor(S0, chance, margin=0.15):
    return (S0 - chance) < margin

def moat_verdict(rows, chance, min_r2=0.8, min_drop=0.15, ratio=2.0):
    # The moat: at matched tokens, the HANDOFF arm loses the fact much faster than the
    # LONG-CONTEXT arm. A FLAT-high long-context curve (fact preserved) is the IDEAL
    # outcome, so we compare DROPS, not long-context fit quality. Passes iff handoff
    # genuinely decays (good fit + real drop) AND drops substantially more than longctx.
    kh, Sh = survival_curve(rows, "handoff")
    kl, Sl = survival_curve(rows, "longctx")
    fh = fit_tau(kh, Sh, chance)
    drop_h = Sh[0] - Sh[-1]
    drop_l = Sl[0] - Sl[-1]
    handoff_decays = (fh["r2"] >= min_r2) and (drop_h >= min_drop)
    passes = bool(handoff_decays and (drop_h - drop_l >= min_drop)
                  and (drop_h >= ratio * max(drop_l, 1e-9)))
    return {"tau_handoff": fh["tau"], "handoff_r2": fh["r2"],
            "drop_handoff": drop_h, "drop_longctx": drop_l,
            "handoff_decays": handoff_decays, "passes": passes}

def bootstrap_survival_ci(rows, condition, n_boot=1000, alpha=0.05, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    ks = sorted({r["k"] for r in rows if r["condition"] == condition})
    out = {}
    for k in ks:
        vals = np.array([r["correct"] for r in rows
                         if r["condition"] == condition and r["k"] == k], float)
        if len(vals) == 0:
            continue
        boots = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                          for _ in range(n_boot)])
        lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
        out[k] = {"mean": float(vals.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(vals))}
    return out

def bootstrap_tau_ci(rows, condition, chance, n_boot=300, alpha=0.05, rng_seed=0):
    # resample FACTS (with replacement), recompute the survival curve, refit tau each draw
    rng = np.random.default_rng(rng_seed)
    sub = [r for r in rows if r["condition"] == condition]
    fact_ids = sorted({r["fact_id"] for r in sub})
    by_fact = {fid: [r for r in sub if r["fact_id"] == fid] for fid in fact_ids}
    ks = sorted({r["k"] for r in sub})
    taus = []
    for _ in range(n_boot):
        draw = rng.choice(fact_ids, size=len(fact_ids), replace=True)
        rows_b = [r for fid in draw for r in by_fact[fid]]
        S = []
        for k in ks:
            vals = [r["correct"] for r in rows_b if r["k"] == k]
            S.append(sum(vals)/len(vals) if vals else 0.0)
        taus.append(fit_tau(ks, S, chance)["tau"])
    taus = np.array(taus)
    return {"tau": float(np.median(taus)),
            "lo": float(np.percentile(taus, 100*alpha/2)),
            "hi": float(np.percentile(taus, 100*(1-alpha/2)))}

def fit_linear(ks, S):
    x = np.asarray(ks, float); y = np.asarray(S, float)
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ [a, b]
    ss_res = float(((y - yhat)**2).sum()); ss_tot = float(((y - y.mean())**2).sum())
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return {"a": float(a), "b": float(b), "r2": r2}

def fit_constant(S):
    y = np.asarray(S, float)
    return {"c": float(y.mean()), "r2": 0.0}

def _aic(rss, n, k_params):
    rss = max(float(rss), 1e-12)
    return n * np.log(rss / n) + 2 * k_params

def _rss(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(((y - yhat)**2).sum())

def compare_decay_models(ks, S, chance):
    ks = np.asarray(ks, float); S = np.asarray(S, float); n = len(ks)
    # exponential with free floor (reuse fit_tau)
    fe = fit_tau(ks, S, chance)
    exp_hat = fe["f"] + (fe["S0"] - fe["f"]) * np.exp(-ks / fe["tau"])
    exp = {"params": {"tau": fe["tau"], "f": fe["f"], "S0": fe["S0"]},
           "r2": fe["r2"], "aic": _aic(_rss(S, exp_hat), n, 3)}
    fl = fit_linear(ks, S)
    lin_hat = fl["a"] + fl["b"] * ks
    lin = {"params": {"a": fl["a"], "b": fl["b"]}, "r2": fl["r2"],
           "aic": _aic(_rss(S, lin_hat), n, 2)}
    fc = fit_constant(S)
    con_hat = np.full_like(ks, fc["c"])
    con = {"params": {"c": fc["c"]}, "r2": fc["r2"], "aic": _aic(_rss(S, con_hat), n, 1)}
    models = {"exponential": exp, "linear": lin, "constant": con}
    best = min(models, key=lambda m: models[m]["aic"])
    return {"models": models, "best": best}

def fit_power_law(sizes, taus):
    x = np.log(np.asarray(sizes, float)); y = np.log(np.asarray(taus, float))
    A = np.vstack([x, np.ones_like(x)]).T
    (alpha, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ [alpha, c]
    ss_res = float(((y-yhat)**2).sum()); ss_tot = float(((y-y.mean())**2).sum()) or 1e-9
    return {"alpha": float(alpha), "c": float(c), "r2": 1 - ss_res/ss_tot}
