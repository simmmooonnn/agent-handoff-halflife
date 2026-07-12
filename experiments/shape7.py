# R1.6 reproducibility (round-5 fix): the 7-point (k=0,1,2,3,4,8,16) exp-vs-stretched
# AICc comparison the doc cites (ΔAICc≈14, s≈1.60) was computed inline and never
# committed. This script recomputes it from committed rows and writes an auditable
# output. S(3) comes from k3_rows (job 18853961); the rest from the pilot ladder.
import json
from statistics import mean
import numpy as np
from scipy.optimize import curve_fit

TAG = "Qwen2.5-7B-Instruct"


def s_at(path, cond, k):
    v = []
    for l in open(f"out/pilot/{path}"):
        r = json.loads(l)
        if r["condition"] == cond and r["k"] == k:
            v.append(r["correct"])
    return mean(v) if v else None


def main():
    ks = [0, 1, 2, 3, 4, 8, 16]
    S = []
    for k in ks:
        if k == 3:
            S.append(s_at(f"k3_rows_{TAG}.jsonl", "handoff", 3))
        else:
            S.append(s_at(f"rows_{TAG}.jsonl", "handoff", k))
    ks_a, S_a = np.array(ks, float), np.array(S, float)

    def aicc(rss, n, p):
        return n * np.log(rss / n) + 2 * p + 2 * p * (p + 1) / (n - p - 1)

    def fexp(k, tau, f, S0):
        return f + (S0 - f) * np.exp(-k / tau)

    def fstr(k, tau, s, f, S0):
        return f + (S0 - f) * np.exp(-((k / tau) ** s))

    pe, _ = curve_fit(fexp, ks_a, S_a, p0=[1.5, 0.22, 0.91], maxfev=40000)
    rss_e = float(np.sum((S_a - fexp(ks_a, *pe)) ** 2))
    ps, _ = curve_fit(fstr, ks_a, S_a, p0=[pe[0], 1.0, pe[1], pe[2]], maxfev=40000)
    rss_s = float(np.sum((S_a - fstr(ks_a, *ps)) ** 2))
    n = len(ks)
    ae, as_ = aicc(rss_e, n, 3), aicc(rss_s, n, 4)
    out = {"grid_ks": ks, "S_handoff": [round(x, 3) for x in S],
           "exp": {"tau": round(pe[0], 3), "f": round(pe[1], 3), "S0": round(pe[2], 3),
                   "rss": rss_e, "aicc": round(ae, 2), "params": 3},
           "stretched_exp": {"tau": round(ps[0], 3), "s": round(ps[1], 3),
                             "f": round(ps[2], 3), "S0": round(ps[3], 3),
                             "rss": rss_s, "aicc": round(as_, 2), "params": 4},
           "delta_aicc_stretched_minus_exp": round(as_ - ae, 2),
           "residual_df_stretched": n - 4,
           "aicc_denominator_note": f"AICc small-sample denom n-k-1 = {n-4-1} for the "
                                    f"4-param fit; ranking is indicative with so few points",
           "preferred": "stretched_exp" if as_ < ae else "exp"}
    json.dump(out, open(f"out/pilot/shape7_{TAG}.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
