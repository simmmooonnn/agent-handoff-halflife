# Cross-family ladder aggregator. The clean (drop-negation) size scaling on Qwen2.5
# is near-null (alpha~0.06, CI includes 0); the robust headline is the budget law.
# This checks whether that size behaviour REPLICATES in a second model family (Phi-3),
# and whether tau is family-invariant at matched size (Qwen ~ Mistral ~ Phi at ~7-14B).
# Pure re-analysis of pilot_gate rows/verdicts; no model calls.
import json, glob, os, sys
import numpy as np
from analyze import fit_tau, fit_power_law

CLEAN_TYPES = {"numeric", "entity", "preference"}   # drop negation (empirical chance ~0.5)

# tag -> (family, N_params)
MODELS = {
    "Qwen2.5-0.5B-Instruct": ("Qwen2.5", 0.5e9), "Qwen2.5-1.5B-Instruct": ("Qwen2.5", 1.5e9),
    "Qwen2.5-3B-Instruct": ("Qwen2.5", 3e9), "Qwen2.5-7B-Instruct": ("Qwen2.5", 7e9),
    "Qwen2.5-14B-Instruct": ("Qwen2.5", 14e9), "Qwen2.5-32B-Instruct": ("Qwen2.5", 32e9),
    "Mistral-7B-Instruct-v0.3": ("Mistral", 7e9),
    "Phi-3-mini-4k-instruct": ("Phi-3", 3.8e9),
    "Phi-3-small-8k-instruct": ("Phi-3", 7e9),
    "Phi-3-medium-4k-instruct": ("Phi-3", 14e9),
}

def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def survival(rows, cond, clean=False):
    sub = [r for r in rows if r["condition"] == cond and (not clean or r["ftype"] in CLEAN_TYPES)]
    ks = sorted({r["k"] for r in sub})
    S = [ (lambda v: sum(v)/len(v) if v else 0.0)([r["correct"] for r in sub if r["k"] == k]) for k in ks ]
    return ks, S

def chance(rows, clean=False):
    v = [r["correct"] for r in rows if r["condition"] == "nofact" and (not clean or r["ftype"] in CLEAN_TYPES)]
    return sum(v)/len(v) if v else 0.0

def boot_tau(rows, cond, clean, ch, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    sub = [r for r in rows if r["condition"] == cond and (not clean or r["ftype"] in CLEAN_TYPES)]
    fids = sorted({r["fact_id"] for r in sub})
    by = {f: [r for r in sub if r["fact_id"] == f] for f in fids}
    ks = sorted({r["k"] for r in sub})
    taus = []
    for _ in range(n_boot):
        draw = rng.choice(fids, size=len(fids), replace=True)
        rb = [r for f in draw for r in by[f]]
        S = [ (lambda v: sum(v)/len(v) if v else 0.0)([r["correct"] for r in rb if r["k"] == k]) for k in ks ]
        taus.append(fit_tau(ks, S, ch)["tau"])
    a = np.array(taus)
    return float(np.median(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def model_taus(tag):
    path = f"out/pilot/rows_{tag}.jsonl"
    if not os.path.exists(path):
        return None
    rows = load_rows(path)
    out = {}
    for clean in (False, True):
        ch = chance(rows, clean)
        ks, S = survival(rows, "handoff", clean)
        fit = fit_tau(ks, S, ch)
        med, lo, hi = boot_tau(rows, "handoff", clean, ch)
        out["clean" if clean else "raw"] = {"tau": fit["tau"], "tau_ci": [lo, hi],
                                            "r2": fit["r2"], "S0": fit["S0"], "chance": ch}
    return out

def main():
    res = {"models": {}, "families": {}}
    print(f"{'model':28}{'family':9}{'N':>8}{'raw_tau':>9}{'clean_tau':>11}{'clean_CI':>16}{'S0':>6}")
    for tag, (fam, N) in sorted(MODELS.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        t = model_taus(tag)
        if t is None:
            continue
        res["models"][tag] = {"family": fam, "N": N, **t}
        ci = t["clean"]["tau_ci"]
        print(f"{tag:28}{fam:9}{N:>8.1e}{t['raw']['tau']:>9.2f}{t['clean']['tau']:>11.2f}"
              f"  [{ci[0]:.2f},{ci[1]:.2f}]{t['raw']['S0']:>6.2f}")
    # per-family power-law fits (raw + clean)
    print("\n=== within-family size scaling  tau ~ N^alpha ===")
    for fam in sorted({m["family"] for m in res["models"].values()}):
        ms = [(m["N"], m) for tag, m in res["models"].items() if m["family"] == fam]
        ms.sort()
        if len(ms) < 2:
            print(f"  {fam:9} only {len(ms)} model(s) -> no within-family slope")
            continue
        Ns = [n for n, _ in ms]
        raw = fit_power_law(Ns, [m["raw"]["tau"] for _, m in ms])
        cln = fit_power_law(Ns, [m["clean"]["tau"] for _, m in ms])
        res["families"][fam] = {"n_models": len(ms), "N": Ns,
                                "alpha_raw": raw["alpha"], "r2_raw": raw["r2"],
                                "alpha_clean": cln["alpha"], "r2_clean": cln["r2"]}
        print(f"  {fam:9} ({len(ms)} models) raw alpha={raw['alpha']:.3f} r2={raw['r2']:.2f} | "
              f"CLEAN alpha={cln['alpha']:.3f} r2={cln['r2']:.2f}")
    # matched-size cross-family invariance (~7B and ~14B bands)
    print("\n=== family-invariance at matched size (clean tau) ===")
    for band, lo, hi in [("~7B", 6e9, 8e9), ("~14B", 13e9, 15e9), ("~3-4B", 3e9, 4e9)]:
        hits = [(m["family"], m["clean"]["tau"], m["clean"]["tau_ci"])
                for m in res["models"].values() if lo <= m["N"] <= hi]
        if len(hits) >= 2:
            desc = ", ".join(f"{f} {t:.2f}[{c[0]:.2f},{c[1]:.2f}]" for f, t, c in hits)
            print(f"  {band}: {desc}")
    json.dump(res, open("out/pilot/cross_family.json", "w"), indent=2)
    print("\nwrote out/pilot/cross_family.json")

if __name__ == "__main__":
    main()
