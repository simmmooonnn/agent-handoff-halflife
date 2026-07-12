# Round-6 cached reanalysis cluster (no new model calls).
#   C9  stretched-exp per budget arm + empirical half-life; refit budget beta on the
#       shape-agnostic empirical half-life and compare to the exp-tau beta (0.82).
#   C11 fixed-k length control: replace the censored tau-ratio "19x" (persona vs
#       neutral_long) with observed Delta-S at k=2 and k=4.
#   C8  variance-weighted (by log-tau CI width) log-log budget beta + cross-family
#       envelope. (Paired-over-facts bootstrap needs Anvil per-fact rows -> flagged.)
#   P6  capacity-arithmetic: fit log tau = c + beta*log B + mu*log M on the cached
#       3x4 grid; compare to ratio-only log tau = c + gamma*log(B/M). Arithmetic null
#       requires beta = -mu. (CIs need Anvil per-fact rows -> flagged.)
import json
import numpy as np
from scipy.optimize import curve_fit

KS6 = np.array([0, 1, 2, 4, 8, 16], float)
KS4 = np.array([0, 1, 2, 4], float)


def load(f):
    return json.load(open(f"out/pilot/{f}.json", encoding="utf-8"))


def stretched(k, f, S0, tau, s):
    return f + (S0 - f) * np.exp(-((k / tau) ** s))


def emp_halflife(ks, S):
    """First k where S crosses 0.5, linear interpolation in k. None if never."""
    S = np.asarray(S, float)
    for i in range(1, len(S)):
        if S[i - 1] >= 0.5 >= S[i]:
            t = (S[i - 1] - 0.5) / (S[i - 1] - S[i] + 1e-12)
            return ks[i - 1] + t * (ks[i] - ks[i - 1])
    return None


def c9():
    bs = load("budget_sweep_Qwen2.5-7B-Instruct")["budgets"]
    rows = []
    for b in bs:
        if b["budget"] == 100:
            continue
        S = np.array(b["S"], float)
        # stretched fit (bounded s)
        try:
            p0 = [b["f"], b["S0"], max(b["tau"], 0.2), 1.0]
            popt, _ = curve_fit(stretched, KS6, S, p0=p0,
                                bounds=([0, 0.5, 0.05, 0.3], [0.6, 1.0, 20, 4]),
                                maxfev=20000)
            s_fit = popt[3]
        except Exception:
            s_fit = float("nan")
        rows.append({"budget": b["budget"], "exp_tau": round(b["tau"], 3),
                     "stretched_s": round(float(s_fit), 3),
                     "emp_halflife_kS0.5": (round(emp_halflife(KS6, S), 3)
                                            if emp_halflife(KS6, S) is not None else None)})
    # beta on exp-tau vs on empirical half-life
    Bs = np.array([r["budget"] for r in rows], float)
    lt_exp = np.log(np.array([r["exp_tau"] for r in rows]))
    hl = np.array([r["emp_halflife_kS0.5"] for r in rows], float)
    beta_exp = np.polyfit(np.log(Bs), lt_exp, 1)[0]
    ok = ~np.isnan(hl)
    beta_hl = (np.polyfit(np.log(Bs[ok]), np.log(hl[ok]), 1)[0] if ok.sum() >= 2 else None)
    return {"rows": rows, "beta_exp_tau": round(float(beta_exp), 3),
            "beta_empirical_halflife": (round(float(beta_hl), 3) if beta_hl else None),
            "note": ("s>1 on most arms -> early decay faster than exp; if beta_hl ~ beta_exp "
                     "the budget law is robust to the shape choice.")}


def c11():
    C = load("framing_sweep_Qwen2.5-7B-Instruct")["conditions"]
    p, n = np.array(C["persona"]["S"], float), np.array(C["neutral_long"]["S"], float)
    # KS4 index: k=2 -> idx2, k=4 -> idx3
    return {"persona_S": list(p), "neutral_long_S": list(n),
            "dS_k2": round(float(n[2] - p[2]), 3), "dS_k4": round(float(n[3] - p[3]), 3),
            "note": ("Observed length-control contrast at matched depth (both directly "
                     "measured) replaces the censored tau-ratio. neutral_long retains far "
                     "more than persona at every k -> prompt-length works OPPOSITE to collapse.")}


def wls_slope(x, y, w):
    # weighted least squares slope of y~x
    W = np.diag(w)
    X = np.vstack([x, np.ones_like(x)]).T
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    return beta[0]


def c8():
    bs = [b for b in load("budget_sweep_Qwen2.5-7B-Instruct")["budgets"] if b["budget"] != 100]
    B = np.array([b["budget"] for b in bs], float)
    tau = np.array([b["tau"] for b in bs], float)
    lo = np.array([b["tau_ci"][0] for b in bs], float)
    hi = np.array([b["tau_ci"][1] for b in bs], float)
    sd_log = (np.log(hi) - np.log(lo)) / (2 * 1.96)      # per-point log-tau sd from CI
    w = 1.0 / sd_log**2
    beta_unw = np.polyfit(np.log(B), np.log(tau), 1)[0]
    beta_w = wls_slope(np.log(B), np.log(tau), w)
    return {"beta_unweighted": round(float(beta_unw), 3),
            "beta_logvar_weighted": round(float(beta_w), 3),
            "per_point_log_tau_sd": [round(float(x), 3) for x in sd_log],
            "cross_family_envelope": {"Phi3-medium": 1.50, "Mistral-7B": 0.38, "Qwen-7B": 0.82},
            "flag": ("point/weighted only; a valid slope CI needs a PAIRED-over-facts "
                     "bootstrap (resample fact indices once, apply to all budget cells) — "
                     "requires the Anvil per-fact verdict rows, not in the local cache.")}


def p6():
    grids = {4: "mxbudget_M4_Qwen2.5-7B-Instruct",
             8: "budget_sweep_Qwen2.5-7B-Instruct",
             16: "mxbudget_M16_Qwen2.5-7B-Instruct"}
    B, M, T = [], [], []
    for m, fn in grids.items():
        for b in load(fn)["budgets"]:
            if b["budget"] == 100:
                continue
            B.append(b["budget"]); M.append(m); T.append(b["tau"])
    B, M, T = np.array(B, float), np.array(M, float), np.array(T, float)
    lt = np.log(T)
    # full model: lt = c + beta logB + mu logM
    X = np.vstack([np.log(B), np.log(M), np.ones_like(B)]).T
    coef, *_ = np.linalg.lstsq(X, lt, rcond=None)
    beta, mu, c = coef
    pred = X @ coef
    r2_full = 1 - np.sum((lt - pred)**2) / np.sum((lt - lt.mean())**2)
    # ratio-only: lt = c + gamma log(B/M)
    Xr = np.vstack([np.log(B / M), np.ones_like(B)]).T
    cr, *_ = np.linalg.lstsq(Xr, lt, rcond=None)
    predr = Xr @ cr
    r2_ratio = 1 - np.sum((lt - predr)**2) / np.sum((lt - lt.mean())**2)
    return {"n_cells": len(B), "beta_B": round(float(beta), 3), "mu_M": round(float(mu), 3),
            "arithmetic_null_needs": "beta == -mu (i.e. tau=f(B/M) only)",
            "beta_plus_mu": round(float(beta + mu), 3),
            "r2_full_BM": round(float(r2_full), 3), "r2_ratio_only": round(float(r2_ratio), 3),
            "flag": ("point estimates only; CIs on beta,mu need a paired-over-facts bootstrap "
                     "of the grid (Anvil per-fact rows). beta+mu far from 0 => not pure "
                     "arithmetic, but state as suggestive until CIs land.")}


if __name__ == "__main__":
    out = {"C9_shape_and_halflife": c9(), "C11_fixed_k_length_control": c11(),
           "C8_beta_weighted_envelope": c8(), "P6_capacity_arithmetic": p6()}
    json.dump(out, open("out/pilot/round6_reanalysis.json", "w"), indent=2)
    for k, v in out.items():
        print(f"=== {k} ===")
        print(json.dumps(v, indent=2))
        print()
