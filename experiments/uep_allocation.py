# Measurement-driven UEP budget allocation (sub-problem; prereg
# out/pilot/uep_allocation_predictions.json, commit 3b080bb). Pure simulation on REAL
# cached per-fact handoff survival (framing_rows_{model}.jsonl); NO API calls.
#
# Given a scarce out-of-band protection budget (protect fraction phi -> those facts
# survive by construction), which facts to offload? Compare allocation policies.
# Calibration split MEASURES per-type survival; held-out TEST split evaluates. Protected
# facts count as survived (out-of-band = value never enters the compressor).
import json, sys
from statistics import mean
import numpy as np

MODELS = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3", "Phi-3-medium-4k-instruct",
          "gpt-5.4-mini", "claude-sonnet-4-6"]
PHIS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
RNG = np.random.default_rng(0)


def load(model):
    rows = [json.loads(l) for l in open(f"out/pilot/framing_rows_{model}.jsonl")]
    h = [r for r in rows if r["condition"] == "handoff"]
    kmax = max(r["k"] for r in h)
    facts = [{"fid": r["fact_id"], "ftype": r["ftype"], "surv": 1 if r["correct"] else 0}
             for r in h if r["k"] == kmax]
    facts.sort(key=lambda f: (f["ftype"], f["fid"]))
    # deterministic split: even index within type = calibration, odd = test
    cal, test = [], []
    seen = {}
    for f in facts:
        i = seen.get(f["ftype"], 0); seen[f["ftype"]] = i + 1
        (cal if i % 2 == 0 else test).append(f)
    return cal, test, kmax


def type_surv(split):
    d = {}
    for t in set(f["ftype"] for f in split):
        v = [f["surv"] for f in split if f["ftype"] == t]
        d[t] = mean(v)
    return d


def evaluate(test, protect_idx):
    """protect_idx: set of test indices protected (survive=1); rest use actual surv."""
    return mean(1 if i in protect_idx else test[i]["surv"] for i in range(len(test)))


def alloc(test, order, b):
    """protect the first b test facts in the given order (list of indices)."""
    return set(order[:b])


def main():
    out = {"models": {}}
    for model in MODELS:
        cal, test, kmax = load(model)
        cal_surv = type_surv(cal)                       # measured signal
        n = len(test)
        # policy orderings over test indices
        idx = list(range(n))
        # measured_greedy: ascending calibration survival of the fact's type
        meas_order = sorted(idx, key=lambda i: (cal_surv.get(test[i]["ftype"], 1.0), test[i]["fid"]))
        # folk_numbers: numeric first, then others (stable by fid)
        folk_order = sorted(idx, key=lambda i: (0 if test[i]["ftype"] == "numeric" else 1, test[i]["fid"]))
        # oracle: dead facts first
        oracle_order = sorted(idx, key=lambda i: (test[i]["surv"], test[i]["fid"]))
        res = {"kmax": kmax, "cal_type_surv": {t: round(v, 3) for t, v in sorted(cal_surv.items())},
               "n_test": n, "curves": {}}
        for phi in PHIS:
            b = round(phi * n)
            # random: average over 200 draws
            rand_vals = []
            for _ in range(200):
                pr = set(RNG.choice(n, size=b, replace=False).tolist()) if b else set()
                rand_vals.append(evaluate(test, pr))
            res["curves"][f"{phi:.1f}"] = {
                "none": round(evaluate(test, set()), 3),
                "random": round(mean(rand_vals), 3),
                "folk_numbers": round(evaluate(test, alloc(test, folk_order, b)), 3),
                "measured_greedy": round(evaluate(test, alloc(test, meas_order, b)), 3),
                "oracle": round(evaluate(test, alloc(test, oracle_order, b)), 3)}
        # value of measurement = area (mean over phi) measured - folk, and measured - random
        phis = [f"{p:.1f}" for p in PHIS]
        res["measured_minus_folk_avg"] = round(mean(res["curves"][p]["measured_greedy"]
                                                    - res["curves"][p]["folk_numbers"] for p in phis), 3)
        res["measured_minus_random_avg"] = round(mean(res["curves"][p]["measured_greedy"]
                                                      - res["curves"][p]["random"] for p in phis), 3)
        res["measured_gap_to_oracle_avg"] = round(mean(res["curves"][p]["oracle"]
                                                       - res["curves"][p]["measured_greedy"] for p in phis), 3)
        out["models"][model] = res

    # cross-model summary
    out["summary"] = {
        "measured_minus_folk_avg": {m: out["models"][m]["measured_minus_folk_avg"] for m in MODELS},
        "measured_minus_random_avg": {m: out["models"][m]["measured_minus_random_avg"] for m in MODELS},
        "measured_gap_to_oracle_avg": {m: out["models"][m]["measured_gap_to_oracle_avg"] for m in MODELS}}
    json.dump(out, open("out/pilot/uep_allocation_results.json", "w"), indent=2)

    print("=== Measurement-driven UEP budget allocation (held-out test survival) ===")
    for m in MODELS:
        r = out["models"][m]
        print(f"\n{m}  (cal type-surv: {r['cal_type_surv']})")
        print(f"  {'phi':>4} | none  rand  folk  MEAS  orac")
        for p in [f"{x:.1f}" for x in PHIS]:
            c = r["curves"][p]
            print(f"  {p:>4} | {c['none']:.2f}  {c['random']:.2f}  {c['folk_numbers']:.2f}  "
                  f"{c['measured_greedy']:.2f}  {c['oracle']:.2f}")
        print(f"  avg: MEAS-folk={r['measured_minus_folk_avg']:+.3f}  "
              f"MEAS-rand={r['measured_minus_random_avg']:+.3f}  "
              f"oracle-MEAS={r['measured_gap_to_oracle_avg']:+.3f}")
    print("\n=== cross-model MEAS - folk (value of measuring beyond 'protect numbers') ===")
    for m in MODELS:
        print(f"  {m:26s}: {out['summary']['measured_minus_folk_avg'][m]:+.3f}")


if __name__ == "__main__":
    main()
