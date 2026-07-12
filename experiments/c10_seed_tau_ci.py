# C10: propagate seed variance into the tau FIT (not just S(k)) for the seedvar
# anchor arms. The surviving headline CIs (budget beta, framing groups) are computed
# greedy-only; here we show, for the arms where we DO have per-fact per-seed rows,
# how much the tau CI widens once sampling variance is folded in via a facts x seed
# variance-components bootstrap. Full propagation to beta and every framing cell needs
# the Anvil per-fact seed rows across all budget/framing cells (flagged).
import json
import numpy as np
from collections import defaultdict

SEED_FILES = [f"out/pilot/seedvar_rows_Qwen2.5-7B-Instruct_s{i}.jsonl" for i in (1, 2, 3)]
RNG = np.random.default_rng(0)
B = 2000


def load_rows():
    # data[cond][fact_id][k] = list of 0/1 over seeds
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ks = set()
    for fn in SEED_FILES:
        for line in open(fn, encoding="utf-8"):
            r = json.loads(line)
            data[r["condition"]][r["fact_id"]][r["k"]].append(1 if r["correct"] else 0)
            ks.add(r["k"])
    return data, sorted(ks)


def fit_tau(ks, S, floor_guess=0.15):
    from scipy.optimize import curve_fit
    ks = np.array(ks, float); S = np.array(S, float)

    def model(k, f, S0, tau):
        return f + (S0 - f) * np.exp(-k / tau)
    try:
        popt, _ = curve_fit(model, ks, S, p0=[floor_guess, S[0], 1.0],
                            bounds=([0, 0.5, 0.05], [0.6, 1.0, 30]), maxfev=10000)
        return popt[2]
    except Exception:
        return np.nan


def boot_tau_ci(data_cond, ks):
    facts = list(data_cond.keys())
    nf = len(facts)
    # per-cond ks: keep only k values present for this arm
    ks = [k for k in ks if any(k in data_cond[f] for f in facts)]
    taus = []
    for _ in range(B):
        idx = RNG.integers(0, nf, nf)                       # integer fact resample
        S = []
        for k in ks:
            vals = [data_cond[facts[j]][k][RNG.integers(len(data_cond[facts[j]][k]))]
                    for j in idx if k in data_cond[facts[j]]]
            S.append(np.mean(vals) if vals else np.nan)
        if any(np.isnan(S)):
            continue
        t = fit_tau(ks, S)
        if not np.isnan(t):
            taus.append(t)
    return (float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5)),
            float(np.median(taus)), ks)


def main():
    data, ks = load_rows()
    out = {"ks": ks, "B": B, "arms": {}}
    for cond in data:
        # greedy-anchor S(k) from the sampling mean (greedy per-fact not in local cache)
        cond_ks = [k for k in ks if any(k in data[cond][f] for f in data[cond])]
        Smean = [np.mean([np.mean(data[cond][f][k]) for f in data[cond] if k in data[cond][f]])
                 for k in cond_ks]
        lo, hi, med, used_ks = boot_tau_ci(data[cond], ks)
        out["arms"][cond] = {"ks": used_ks, "sampling_mean_S": [round(x, 3) for x in Smean],
                             "tau_median_seedinflated": round(med, 3),
                             "tau_ci_seedinflated": [round(lo, 3), round(hi, 3)]}
    out["flag"] = ("CI reflects facts x 3-sampling-seed variance for these anchor arms; "
                   "greedy per-fact rows not in local cache, so this is the fully-sampled "
                   "band (mildly wider than greedy+seed). Propagating to the budget beta and "
                   "every framing cell needs Anvil per-fact seed rows across all cells.")
    json.dump(out, open("out/pilot/c10_seed_tau_ci.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
