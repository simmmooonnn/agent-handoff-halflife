# Item 7 (round-5, domain reviewer): the prediction/predictability literature
# (PredictaBoard etc.) scores a probability forecast with Brier + calibration +
# a skill score, not just median|err|. We forecast a per-arm survival PROBABILITY
# (S(4)_probe); here we score it at the per-FACT (instance) level: for each fact,
# forecast = its arm's probe probability, outcome = did that fact survive at k=4.
#   Brier      = mean over facts of (p - y)^2   (lower better)
#   Skill      = 1 - Brier / Brier_baseline     (baseline = predict global base rate)
#   ECE        = calibration error over probability bins
# Reported for the 2-hop probe forecast and, as reference, the length-only null.
import json
from statistics import mean
FLOOR = 0.22158
TAG = "Qwen2.5-7B-Instruct"


def outcomes_at4(rows, lab, labfield=None):
    return [r["correct"] for r in rows
            if (r.get(labfield) if labfield else r["condition"]) == lab and r["k"] == 4]


def main():
    rep = json.load(open(f"out/pilot/probe_report_{TAG}.json"))
    fr = [json.loads(l) for l in open(f"out/pilot/framing_rows_{TAG}.jsonl")]
    pr = [json.loads(l) for l in open(f"out/pilot/protocol_rows_{TAG}.jsonl")]
    ns = [json.loads(l) for l in open(f"out/pilot/newstance_rows_{TAG}.jsonl")]
    for r in ns:
        r["_lab"] = r.get("label") or r["condition"]
    SRC = {"selfsumm": (fr, None), "persona": (fr, None), "node": (fr, None),
           "rolecont": (fr, None), "distrust": (fr, None), "neutral_long": (fr, None),
           "faithful": (fr, None), "itemize": (pr, None), "manifest": (pr, None),
           "faithman": (pr, None), "ledger@25": (ns, "_lab"),
           "editor@25": (ns, "_lab"), "link@25": (ns, "_lab")}

    ys, p_probe = [], []
    for a in rep["arms"]:
        arm = a["arm"]
        rows, lf = SRC[arm]
        outs = outcomes_at4(rows, arm, lf)
        ys += outs
        p_probe += [a["S4_probe"]] * len(outs)
    base = mean(ys)
    brier_probe = mean((p - y) ** 2 for p, y in zip(p_probe, ys))
    brier_base = mean((base - y) ** 2 for y in ys)
    skill = 1 - brier_probe / brier_base if brier_base else None

    # ECE over 5 bins on the probe forecast
    bins = [[] for _ in range(5)]
    for p, y in zip(p_probe, ys):
        bins[min(4, int(p * 5))].append((p, y))
    ece = 0.0
    rel = []
    for b in bins:
        if not b:
            continue
        conf = mean(p for p, _ in b); acc = mean(y for _, y in b)
        ece += len(b) / len(ys) * abs(conf - acc)
        rel.append({"conf": round(conf, 3), "obs": round(acc, 3), "n": len(b)})
    out = {"n_instances": len(ys), "base_rate": round(base, 3),
           "brier_probe": round(brier_probe, 4), "brier_baserate": round(brier_base, 4),
           "skill_score": round(skill, 3), "ECE": round(ece, 3), "reliability": rel,
           "note": "PredictaBoard-style per-instance scoring of the 2-hop probe as a "
                   "survival-probability forecast. Skill>0 means it beats predicting the "
                   "global base rate; ECE is mean calibration gap."}
    json.dump(out, open(f"out/pilot/brier_{TAG}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "reliability"}, indent=1))
    for r in rel:
        print(f"  bin conf={r['conf']:.2f} obs={r['obs']:.2f} n={r['n']}")


if __name__ == "__main__":
    main()
