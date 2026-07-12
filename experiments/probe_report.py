# Item 9b (round-5): emit the 2-hop probe forecast errors to an auditable JSON,
# split by arm class, instead of computing them inside plotting code (figures.py).
# Makes the "median err 0.027 / 0.005" numbers reproducible and shows WHICH arms
# enter each median (low-floor collapse arms vs high-floor preservation arms).
import json
from statistics import mean, median

FLOOR = 0.22158
TAG = "Qwen2.5-7B-Instruct"


def srv(rows, lab, k, labfield=None):
    v = []
    for r in rows:
        key = r.get(labfield) if labfield else r["condition"]
        if key == lab and r["k"] == k:
            v.append(r["correct"])
    return mean(v) if v else None


def probe(s1, s2):
    if s1 <= FLOOR or s2 <= FLOOR:
        return max(s2, 0.0)
    return FLOOR + (s2 - FLOOR) * ((s2 - FLOOR) / (s1 - FLOOR)) ** 2


def main():
    fr = [json.loads(l) for l in open(f"out/pilot/framing_rows_{TAG}.jsonl")]
    pr = [json.loads(l) for l in open(f"out/pilot/protocol_rows_{TAG}.jsonl")]
    ns = [json.loads(l) for l in open(f"out/pilot/newstance_rows_{TAG}.jsonl")]
    for r in ns:
        r["_lab"] = r.get("label") or r["condition"]

    TRAIN = [("selfsumm", fr, None), ("persona", fr, None), ("node", fr, None),
             ("rolecont", fr, None), ("distrust", fr, None), ("neutral_long", fr, None),
             ("faithful", fr, None), ("itemize", pr, None), ("manifest", pr, None),
             ("faithman", pr, None)]
    NEW = [("ledger@25", ns, "_lab"), ("editor@25", ns, "_lab"),
           ("link@25", ns, "_lab")]

    def one(lab, rows, lf):
        s1, s2, s4 = (srv(rows, lab, k, lf) for k in (1, 2, 4))
        if None in (s1, s2, s4):
            return None
        p = probe(s1, s2)
        return {"arm": lab, "S1": round(s1, 3), "S2": round(s2, 3),
                "S4_obs": round(s4, 3), "S4_probe": round(p, 3),
                "abs_err": round(abs(p - s4), 3),
                "class": "high_floor_preservation" if s4 > 0.35 else "low_floor_collapse"}

    train = [r for r in (one(*a) for a in TRAIN) if r]
    new = [r for r in (one(*a) for a in NEW) if r]
    all_arms = train + new
    low = [r["abs_err"] for r in all_arms if r["class"] == "low_floor_collapse"]
    high = [r["abs_err"] for r in all_arms if r["class"] == "high_floor_preservation"]
    out = {"floor": FLOOR, "arms": all_arms,
           "median_err_all_train": round(median([r["abs_err"] for r in train]), 3),
           "median_err_new_stances": round(median([r["abs_err"] for r in new]), 3),
           "median_err_low_floor_collapse": round(median(low), 3) if low else None,
           "median_err_high_floor_preservation": round(median(high), 3) if high else None,
           "note": "2-hop probe is accurate on low-floor collapse arms (already near "
                   "chance) and mis-predicts high-floor preservation arms (private floor "
                   "> global floor); the latter need a deeper (k~4) probe."}
    json.dump(out, open(f"out/pilot/probe_report_{TAG}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
    for r in all_arms:
        print(f"  {r['arm']:>15} S4obs={r['S4_obs']:.2f} probe={r['S4_probe']:.2f} "
              f"err={r['abs_err']:.3f}  [{r['class']}]")


if __name__ == "__main__":
    main()
