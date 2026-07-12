# Per-ftype tau fits from cached rows (reviewer R1-C2: mixing 4 heterogeneous fact
# types -- especially negation with empirical chance ~0.5 -- into ONE exponential
# contaminates tau, f, and the AIC shape verdict). Refit tau separately per fact
# type with per-type empirical chance + bootstrap CI. This also produces the H3
# "stickiness" ranking for free. Pure re-analysis; no model calls.
import json, sys, glob, os
import numpy as np
from analyze import fit_tau

def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def survival(rows, cond, ftype=None):
    sub = [r for r in rows if r["condition"] == cond and (ftype is None or r["ftype"] == ftype)]
    ks = sorted({r["k"] for r in sub})
    S = []
    for k in ks:
        v = [r["correct"] for r in sub if r["k"] == k]
        S.append(sum(v) / len(v) if v else 0.0)
    return ks, S

def emp_chance(rows, ftype=None):
    v = [r["correct"] for r in rows if r["condition"] == "nofact" and (ftype is None or r["ftype"] == ftype)]
    return sum(v) / len(v) if v else 0.0

def boot_tau(rows, cond, ftype, chance, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    sub = [r for r in rows if r["condition"] == cond and r["ftype"] == ftype]
    fids = sorted({r["fact_id"] for r in sub})
    by = {f: [r for r in sub if r["fact_id"] == f] for f in fids}
    ks = sorted({r["k"] for r in sub})
    taus = []
    for _ in range(n_boot):
        draw = rng.choice(fids, size=len(fids), replace=True)
        rb = [r for f in draw for r in by[f]]
        S = []
        for k in ks:
            vv = [r["correct"] for r in rb if r["k"] == k]
            S.append(sum(vv) / len(vv) if vv else 0.0)
        taus.append(fit_tau(ks, S, chance)["tau"])
    taus = np.array(taus)
    return float(np.median(taus)), float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))

def main(tag="Qwen2.5-7B-Instruct", cond="handoff"):
    path = f"out/pilot/rows_{tag}.jsonl"
    rows = load_rows(path)
    ftypes = sorted({r["ftype"] for r in rows})
    print(f"=== {tag}  condition={cond} ===")
    # aggregate (the contaminated fit, for comparison)
    ks, S = survival(rows, cond)
    ch = emp_chance(rows)
    agg = fit_tau(ks, S, ch)
    print(f"AGGREGATE (mixed 4 ftypes): tau={agg['tau']:.2f} f={agg['f']:.2f} r2={agg['r2']:.2f} chance={ch:.2f}")
    print(f"   S={[round(x,2) for x in S]}")
    print("--- per-ftype (the honest fits + H3 stickiness) ---")
    out = {"tag": tag, "cond": cond, "aggregate": {"tau": agg["tau"], "f": agg["f"], "r2": agg["r2"], "chance": ch, "S": S},
           "by_ftype": {}}
    rank = []
    for ft in ftypes:
        ks_t, S_t = survival(rows, cond, ft)
        ch_t = emp_chance(rows, ft)
        fit = fit_tau(ks_t, S_t, ch_t)
        med, lo, hi = boot_tau(rows, cond, ft, ch_t)
        out["by_ftype"][ft] = {"tau": fit["tau"], "tau_ci": [lo, hi], "f": fit["f"],
                               "r2": fit["r2"], "chance": ch_t, "S": S_t}
        rank.append((fit["tau"], ft))
        print(f"  {ft:10s} tau={fit['tau']:.2f} CI=[{lo:.2f},{hi:.2f}] f={fit['f']:.2f} "
              f"r2={fit['r2']:.2f} chance={ch_t:.2f} S={[round(x,2) for x in S_t]}")
    rank.sort(reverse=True)
    print("H3 stickiness (high tau = stickier): " + " > ".join(f"{ft}({t:.2f})" for t, ft in rank))
    os.makedirs("out/pilot", exist_ok=True)
    json.dump(out, open(f"out/pilot/perftype_{tag}_{cond}.json", "w"), indent=2)

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
