# Tier-1 #4: statistical hardening the reviewers asked for.
#  (A) bootstrap slope CI for the SIZE exponent alpha -- raw (all fact types) vs
#      CLEAN (negation dropped), to substantiate "alpha ~ 0.06 clean, ~0.16 raw is a
#      negation confound". Rigorous nested fact-resampling (raw rows available).
#  (B) slope CI for the BUDGET exponent beta -- parametric bootstrap propagating the
#      per-budget tau CIs already computed (raw budget rows not kept locally).
#  (C) AICc decay-shape model comparison on the 7B handoff curve, adding the
#      stretched-exponential and power-law competitors the reviewers named
#      (geometric == free-floor exponential, noted, not re-fit).
import json, sys
import numpy as np
from scipy.optimize import curve_fit
from analyze import load_rows, fit_tau, fit_power_law, fit_linear, fit_constant, _rss, _aic

SIZES = {"0.5B": 0.5e9, "1.5B": 1.5e9, "3B": 3e9, "7B": 7e9, "14B": 14e9, "32B": 32e9}
LADDER = {t: f"out/pilot/rows_Qwen2.5-{t}-Instruct.jsonl" for t in SIZES}


def _tau_from_rows(rows, exclude_ftypes=()):
    hh = [r for r in rows if r["condition"] == "handoff" and r["ftype"] not in exclude_ftypes]
    nf = [r for r in rows if r["condition"] == "nofact" and r["ftype"] not in exclude_ftypes]
    chance = (sum(r["correct"] for r in nf) / len(nf)) if nf else 0.0
    ks = sorted({r["k"] for r in hh})
    S = [np.mean([r["correct"] for r in hh if r["k"] == k]) for k in ks]
    return fit_tau(ks, S, chance)["tau"]


def bootstrap_alpha(exclude_ftypes=(), n_boot=500, seed=0):
    per_rows = {t: load_rows(p) for t, p in LADDER.items()}
    fids = {t: sorted({r["fact_id"] for r in rows}) for t, rows in per_rows.items()}
    byfact = {t: {fid: [r for r in per_rows[t] if r["fact_id"] == fid] for fid in fids[t]}
              for t in LADDER}
    sizes = [SIZES[t] for t in LADDER]
    rng = np.random.default_rng(seed)
    point_taus = [_tau_from_rows(per_rows[t], exclude_ftypes) for t in LADDER]
    point_alpha = fit_power_law(sizes, point_taus)["alpha"]
    alphas = []
    for _ in range(n_boot):
        taus = []
        for t in LADDER:
            draw = rng.choice(fids[t], size=len(fids[t]), replace=True)
            rows_b = [r for fid in draw for r in byfact[t][fid]]
            taus.append(_tau_from_rows(rows_b, exclude_ftypes))
        alphas.append(fit_power_law(sizes, taus)["alpha"])
    a = np.array(alphas)
    return {"alpha_point": point_alpha, "alpha_boot_median": float(np.median(a)),
            "ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "taus_point": dict(zip(LADDER, [round(x, 3) for x in point_taus]))}


def bootstrap_beta(path="out/pilot/budget_sweep_Qwen2.5-7B-Instruct.json",
                   n_boot=5000, seed=0, max_budget=50):
    d = json.load(open(path))
    bs = [b for b in d["budgets"] if b["budget"] <= max_budget]
    budgets = [b["budget"] for b in bs]
    tau = [b["tau"] for b in bs]
    # lognormal sd from each budget's percentile CI: (ln hi - ln lo)/(2*1.96)
    sd = [max((np.log(b["tau_ci"][1]) - np.log(b["tau_ci"][0])) / (2 * 1.959964), 1e-6) for b in bs]
    mu = [np.log(t) for t in tau]
    rng = np.random.default_rng(seed)
    betas = []
    for _ in range(n_boot):
        samp = [np.exp(rng.normal(m, s)) for m, s in zip(mu, sd)]
        betas.append(fit_power_law(budgets, samp)["beta" if False else "alpha"])
    b = np.array(betas)
    return {"beta_point": fit_power_law(budgets, tau)["alpha"],
            "beta_boot_median": float(np.median(b)),
            "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "budgets": budgets, "tau": [round(x, 3) for x in tau]}


def _aicc(aic, n, k):
    denom = n - k - 1
    return aic + (2 * k * (k + 1) / denom if denom > 0 else float("inf"))


def decay_model_comparison(model_tag="7B"):
    rows = load_rows(LADDER[model_tag])
    hh = [r for r in rows if r["condition"] == "handoff"]
    nf = [r for r in rows if r["condition"] == "nofact"]
    chance = sum(r["correct"] for r in nf) / len(nf)
    ks = np.array(sorted({r["k"] for r in hh}), float)
    S = np.array([np.mean([r["correct"] for r in hh if r["k"] == k]) for k in ks])
    n = len(ks)
    out = {}

    # exponential free-floor (== geometric r^k, r=exp(-1/tau)); 3 params
    fe = fit_tau(ks, S, chance)
    ehat = fe["f"] + (fe["S0"] - fe["f"]) * np.exp(-ks / fe["tau"])
    out["exponential(=geometric)"] = {"k": 3, "r2": fe["r2"], "rss": _rss(S, ehat)}

    # stretched exponential: f+(S0-f)exp(-(k/tau)^s); 4 params
    def _stretch(k, tau, s, f, S0):
        return f + (S0 - f) * np.exp(-(np.asarray(k, float) / tau) ** s)
    try:
        p, _ = curve_fit(_stretch, ks, S, p0=[fe["tau"], 1.0, fe["f"], fe["S0"]],
                         maxfev=40000, bounds=([1e-3, 0.2, -0.2, 0.0], [1e6, 5.0, 1.0, 1.0001]))
        shat = _stretch(ks, *p)
        out["stretched_exp"] = {"k": 4, "params": {"tau": float(p[0]), "s": float(p[1])},
                                "rss": _rss(S, shat)}
    except Exception as e:
        out["stretched_exp"] = {"k": 4, "err": str(e)}

    # power-law decay: f+(S0-f)(1+k)^(-g); 3 params
    def _power(k, g, f, S0):
        return f + (S0 - f) * (1.0 + np.asarray(k, float)) ** (-g)
    try:
        p, _ = curve_fit(_power, ks, S, p0=[1.0, fe["f"], fe["S0"]],
                         maxfev=40000, bounds=([1e-3, -0.2, 0.0], [20.0, 1.0, 1.0001]))
        phat = _power(ks, *p)
        out["power_law"] = {"k": 3, "params": {"gamma": float(p[0])}, "rss": _rss(S, phat)}
    except Exception as e:
        out["power_law"] = {"k": 3, "err": str(e)}

    fl = fit_linear(ks, S)
    out["linear"] = {"k": 2, "r2": fl["r2"], "rss": _rss(S, fl["a"] + fl["b"] * ks)}
    fc = fit_constant(S)
    out["constant"] = {"k": 1, "rss": _rss(S, np.full_like(ks, fc["c"]))}

    for m, v in out.items():
        if "rss" in v:
            v["aic"] = _aic(v["rss"], n, v["k"])
            v["aicc"] = _aicc(v["aic"], n, v["k"])
    ok = {m: v for m, v in out.items() if "aicc" in v}
    best = min(ok, key=lambda m: ok[m]["aicc"])
    best_aicc = ok[best]["aicc"]
    for m in ok:
        ok[m]["delta_aicc"] = ok[m]["aicc"] - best_aicc
    return {"ks": list(ks), "S": [round(x, 3) for x in S], "chance": round(chance, 3),
            "models": out, "best": best}


def main():
    print("=== (A) SIZE exponent alpha, bootstrap over facts ===")
    raw = bootstrap_alpha(exclude_ftypes=())
    clean = bootstrap_alpha(exclude_ftypes=("negation",))
    print(f"  raw   (all ftypes): alpha={raw['alpha_point']:+.3f}  boot_med={raw['alpha_boot_median']:+.3f}  95%CI=[{raw['ci'][0]:+.3f},{raw['ci'][1]:+.3f}]")
    print(f"        per-size tau: {raw['taus_point']}")
    print(f"  clean (no negation): alpha={clean['alpha_point']:+.3f}  boot_med={clean['alpha_boot_median']:+.3f}  95%CI=[{clean['ci'][0]:+.3f},{clean['ci'][1]:+.3f}]")
    print(f"        per-size tau: {clean['taus_point']}")
    raw_sig = "SIGNIFICANT (excludes 0)" if raw['ci'][0] > 0 else "NOT significant (CI includes 0)"
    clean_sig = "SIGNIFICANT (excludes 0)" if clean['ci'][0] > 0 else "NOT significant (CI includes 0)"
    print(f"  -> raw slope {raw_sig}; clean slope {clean_sig}")

    print("\n=== (B) BUDGET exponent beta, parametric bootstrap over per-budget tau CIs ===")
    bb = bootstrap_beta()
    print(f"  budgets={bb['budgets']}  tau={bb['tau']}")
    print(f"  beta={bb['beta_point']:+.3f}  boot_med={bb['beta_boot_median']:+.3f}  95%CI=[{bb['ci'][0]:+.3f},{bb['ci'][1]:+.3f}]")
    print(f"  -> {'SIGNIFICANT (excludes 0)' if bb['ci'][0] > 0 else 'NOT significant'}")

    print("\n=== (C) decay-shape AICc comparison (7B handoff) ===")
    dc = decay_model_comparison("7B")
    print(f"  ks={dc['ks']}  S={dc['S']}  chance={dc['chance']}")
    for m, v in sorted(dc["models"].items(), key=lambda kv: kv[1].get("aicc", 9e9)):
        if "aicc" in v:
            print(f"  {m:24s} k={v['k']} r2={v.get('r2', float('nan')):.3f} "
                  f"AICc={v['aicc']:8.2f} dAICc={v['delta_aicc']:6.2f}"
                  + (f"  {v['params']}" if 'params' in v else ""))
        else:
            print(f"  {m:24s} FAILED: {v.get('err')}")
    print(f"  BEST by AICc: {dc['best']}")

    out = {"alpha_raw": raw, "alpha_clean": clean, "beta": bb, "decay_shape_7B": dc}
    json.dump(out, open("out/pilot/tier1_stats.json", "w"), indent=2)
    print("\nwrote out/pilot/tier1_stats.json")


if __name__ == "__main__":
    main()
