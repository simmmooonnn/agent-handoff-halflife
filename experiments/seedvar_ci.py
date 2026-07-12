# Item 10 (round-5): headline CIs currently bootstrap over FACTS along a single
# greedy trajectory, ignoring decoding stochasticity. This folds seed variance in
# via a variance-components bootstrap: resample facts AND, for each resampled fact,
# resample which of the {greedy, seed1, seed2, seed3} trajectories is observed.
# Reports the widened S(k) CI vs the greedy-only CI for the 4 seedvar anchor arms.
import json
from statistics import mean
import numpy as np

TAG = "Qwen2.5-7B-Instruct"
CELLS = [("handoff", [0, 1, 2, 4, 8]), ("manifest", [0, 1, 2, 4, 8]),
         ("ledger", [0, 1, 2, 4, 8]), ("node", [0, 1, 2, 4])]


def per_fact_outcomes(cond, k):
    # returns {fact_id: [outcome_greedy, outcome_s1, outcome_s2, outcome_s3]}
    src = {}
    # greedy source: framing/protocol/newstance rows
    greedy_file = {"handoff": f"framing_rows_{TAG}.jsonl",
                   "node": f"framing_rows_{TAG}.jsonl",
                   "manifest": f"protocol_rows_{TAG}.jsonl",
                   "ledger": f"newstance_rows_{TAG}.jsonl"}[cond]
    lab = {"ledger": "ledger@25"}.get(cond)
    for l in open(f"out/pilot/{greedy_file}"):
        r = json.loads(l)
        key = r.get("label") if lab else r["condition"]
        if key == (lab or cond) and r["k"] == k:
            src.setdefault(r["fact_id"], [None, None, None, None])[0] = r["correct"]
    for si, s in enumerate((1, 2, 3), start=1):
        for l in open(f"out/pilot/seedvar_rows_{TAG}_s{s}.jsonl"):
            r = json.loads(l)
            if r["condition"] == cond and r["k"] == k:
                if r["fact_id"] in src:
                    src[r["fact_id"]][si] = r["correct"]
    return {fid: [o for o in outs if o is not None] for fid, outs in src.items()}


def main():
    rng = np.random.default_rng(0)
    out = {"note": "variance-components bootstrap: resample facts x trajectory "
                   "(greedy+3 seeds). Widened CI vs greedy-only.", "cells": {}}
    for cond, ks in CELLS:
        for k in ks:
            pf = per_fact_outcomes(cond, k)
            fids = [f for f in pf if pf[f]]
            if not fids:
                continue
            greedy_S = mean(pf[f][0] for f in fids)
            # greedy-only bootstrap over facts
            g = [mean(pf[f][0] for f in rng.choice(fids, len(fids)))
                 for _ in range(1500)]
            # variance-components: resample facts, then a random trajectory per fact
            vc = []
            for _ in range(1500):
                samp = rng.choice(fids, len(fids))
                vc.append(mean(pf[f][rng.integers(0, len(pf[f]))] for f in samp))
            out["cells"][f"{cond}@k{k}"] = {
                "greedy_S": round(greedy_S, 3),
                "ci_greedy_only": [round(float(np.percentile(g, 2.5)), 3),
                                   round(float(np.percentile(g, 97.5)), 3)],
                "ci_with_seed_variance": [round(float(np.percentile(vc, 2.5)), 3),
                                          round(float(np.percentile(vc, 97.5)), 3)],
                "n_traj_per_fact": len(pf[fids[0]])}
    json.dump(out, open(f"out/pilot/seedvar_ci_{TAG}.json", "w"), indent=2)
    for c, v in out["cells"].items():
        gw = v["ci_greedy_only"][1] - v["ci_greedy_only"][0]
        vw = v["ci_with_seed_variance"][1] - v["ci_with_seed_variance"][0]
        print(f"  {c:>14} S={v['greedy_S']:.2f}  greedy CI w={gw:.3f}  "
              f"+seed CI w={vw:.3f}  (x{vw/gw:.2f})" if gw else f"  {c}")


if __name__ == "__main__":
    main()
