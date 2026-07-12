# Item 6 (round-5, addresses C5 "method is Qwen-7B-only"): does the two-dial
# decomposition generalize? Specifically, does dial 2 (the preservation bonus above
# the length curve) EXIST on other families? Cached first pass (no new GPU):
# fit the neutral length curve from each family's budget anchors, then check
# whether manifest/faithful sit ABOVE it (bonus > 0) at matched realized length.
# Run on Anvil (caches live there). Full held-out/newstance cross-family is GPU
# work; this establishes whether the PHENOMENON (dial 2) replicates.
import hashlib, json, sys
from statistics import mean
import numpy as np
from facts import make_facts
from relay import item_id_for, TEMPLATE_VERSION
from run import FILLER

FRAMING_CONDS = ["manifest", "faithful"]  # preservation arms present in xfam framing runs


def realized_w(cache, prefix, facts, by_type, cond, budget, m_facts=8, ks=(1, 2, 4)):
    lens = []
    for f in facts:
        pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
        distractors = tuple(pool[:m_facts - 1])
        dctx = ",".join(d.fact_id for d in distractors)
        ctx = hashlib.sha1(f"{FILLER}|{TEMPLATE_VERSION}|{dctx}".encode()).hexdigest()[:8]
        for k in ks:
            iid = item_id_for(f, k, cond, budget, "light", 0)
            for i in range(k):
                key = f"{prefix}:{iid}:{ctx}:h{i}"
                if key in cache:
                    lens.append(len(cache[key].split()))
    return mean(lens) if lens else float("nan")


def s4(rows_path, cond):
    v = [json.loads(l)["correct"] for l in open(rows_path)
         if json.loads(l)["condition"] == cond and json.loads(l)["k"] == 4]
    # re-read cleanly
    v = []
    for l in open(rows_path):
        r = json.loads(l)
        if r["condition"] == cond and r["k"] == 4:
            v.append(r["correct"])
    return mean(v) if v else None


def run_family(model):
    tag = model.split("/")[-1]
    prefix = f"hf/{tag}"
    cache = json.load(open(f"data/cache_pilot_{tag}.json"))
    facts = make_facts(100, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    bs = json.load(open(f"out/pilot/budget_sweep_{tag}.json"))
    anchors = []
    for b in bs["budgets"]:
        if b["budget"] > 50:
            continue
        w = realized_w(cache, prefix, facts, by_type, "handoff", b["budget"])
        anchors.append({"budget": b["budget"], "w": w, "tau": b["tau"], "f": b["f"]})
    ws = np.array([a["w"] for a in anchors]); taus = np.array([a["tau"] for a in anchors])
    slope, logc = np.polyfit(np.log(ws), np.log(taus), 1)
    fbar = float(np.mean([a["f"] for a in anchors])); S0 = 0.91

    def s4_pred(w):
        tau = np.exp(logc) * w ** slope
        return fbar + (S0 - fbar) * np.exp(-4.0 / tau)

    fr = f"out/pilot/framing_rows_{tag}.jsonl"
    arms = []
    for cond in FRAMING_CONDS:
        obs = s4(fr, cond)
        if obs is None:
            continue
        w = realized_w(cache, prefix, facts, by_type, cond, 25)
        pred = float(s4_pred(w))
        arms.append({"arm": f"{cond}@25", "realized_w": round(w, 1),
                     "S4_obs": round(obs, 3), "S4_pred_from_length": round(pred, 3),
                     "stance_residual": round(obs - pred, 3),
                     "dial2_present": bool(obs - pred > 0.10)})
    return {"model": model, "curve": {"c": float(np.exp(logc)), "slope": float(slope),
                                      "floor": fbar}, "anchors": anchors, "arms": arms}


def main(*models):
    models = list(models) or ["mistralai/Mistral-7B-Instruct-v0.3",
                              "microsoft/Phi-3-medium-4k-instruct"]
    out = {}
    for m in models:
        r = run_family(m)
        out[m.split("/")[-1]] = r
        print(f"\n=== {m} ===  length curve tau={r['curve']['c']:.3f}*w^{r['curve']['slope']:.2f} floor={r['curve']['floor']:.2f}")
        for a in r["arms"]:
            print(f"  {a['arm']:>12} realized={a['realized_w']}w S4_obs={a['S4_obs']} "
                  f"pred(len)={a['S4_pred_from_length']} resid={a['stance_residual']:+.3f} "
                  f"dial2={'YES' if a['dial2_present'] else 'no'}")
    json.dump(out, open("out/pilot/crossfam_surface.json", "w"), indent=2)
    print("\nwrote out/pilot/crossfam_surface.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
