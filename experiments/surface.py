# The instruction surface: decompose every measured arm's S(4) into
#   (1) the REALIZED-LENGTH effect (the budget law evaluated at realized words), and
#   (2) a STANCE residual (above/below the neutral length curve).
# Motivation (proto2): survival tracks realized carry length (itemize@12 ~ faithful@50
# ~ faithful@25 at ~21-28w; itemize@25 at 42w -> 0.87), i.e. the itemize "format
# effect" is mediated by length; but faithful@25 (20.7w, 0.58) vs handoff@25
# (15.8w, 0.23) is far too large a gap for 5 words -> stance has a direct effect.
# This script quantifies both dials for every arm we measured, from cached data only.
import hashlib, json, sys
from statistics import mean
import numpy as np
from facts import make_facts
from relay import item_id_for, TEMPLATE_VERSION
from run import FILLER

TAG = "Qwen2.5-7B-Instruct"
CHANCE = 0.13

def hop_lengths(cache, facts, by_type, cond, budget, m_facts=8, ks=(1, 2, 4)):
    prefix = f"hf/{TAG}"
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
    return lens

def s4_from_rows(path, cond, label_field=None, label=None):
    vals = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["k"] != 4:
            continue
        if label_field:
            if r.get(label_field) == label:
                vals.append(r["correct"])
        elif r["condition"] == cond:
            vals.append(r["correct"])
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

def main():
    cache = json.load(open(f"data/cache_pilot_{TAG}.json", encoding="utf-8"))
    facts = make_facts(100, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)

    # --- anchors: neutral handoff at budgets 8/16/25/50 (the pure length curve) ---
    bs = json.load(open(f"out/pilot/budget_sweep_{TAG}.json"))
    anchors = []
    for b in bs["budgets"]:
        if b["budget"] > 50:
            continue
        w = mean(hop_lengths(cache, facts, by_type, "handoff", b["budget"]))
        anchors.append({"budget": b["budget"], "w": w, "S4": b["S"][3],
                        "tau": b["tau"], "f": b["f"]})
    ws = np.array([a["w"] for a in anchors]); taus = np.array([a["tau"] for a in anchors])
    slope, logc = np.polyfit(np.log(ws), np.log(taus), 1)
    fbar = float(np.mean([a["f"] for a in anchors])); S0 = 0.91
    def s4_pred(w):
        tau = np.exp(logc) * w ** slope
        return fbar + (S0 - fbar) * np.exp(-4.0 / tau)
    print(f"length curve from neutral anchors: tau = {np.exp(logc):.3f} * w^{slope:.2f}, "
          f"floor={fbar:.2f}")
    for a in anchors:
        print(f"  anchor handoff@{a['budget']}: realized={a['w']:.1f}w S4={a['S4']:.2f} "
              f"pred={s4_pred(a['w']):.2f}")

    # --- all arms: (source_rows, cond, budget, optional label) ---
    FR = f"out/pilot/framing_rows_{TAG}.jsonl"
    PR = f"out/pilot/protocol_rows_{TAG}.jsonl"
    P2 = f"out/pilot/proto2_rows_{TAG}.jsonl"
    arms = [
        (FR, "selfsumm", 25, None), (FR, "persona", 25, None), (FR, "node", 25, None),
        (FR, "rolecont", 25, None), (FR, "distrust", 25, None),
        (FR, "neutral_long", 25, None), (FR, "faithful", 25, None),
        (PR, "itemize", 25, None), (PR, "manifest", 25, None), (PR, "faithman", 25, None),
        (P2, "itemize", 12, "itemize@12"), (P2, "faithful", 50, "faithful@50"),
    ]
    out = {"anchors": anchors, "curve": {"c": float(np.exp(logc)), "slope": float(slope),
                                         "floor": fbar}, "arms": []}
    rows_fmt = []
    for path, cond, budget, label in arms:
        if label:
            s4, nn = s4_from_rows(path, cond, "label", label)
        else:
            s4, nn = s4_from_rows(path, cond)
        if s4 is None:
            continue
        lens = hop_lengths(cache, facts, by_type, cond, budget)
        w = mean(lens) if lens else float("nan")
        pred = float(s4_pred(w))
        resid = s4 - pred
        name = label or f"{cond}@{budget}"
        out["arms"].append({"arm": name, "realized_w": w, "S4": s4, "n": nn,
                            "S4_pred_from_length": pred, "stance_residual": resid})
        rows_fmt.append((resid, name, w, s4, pred))
    rows_fmt.sort()
    print(f"\n{'arm':>15} {'real_w':>7} {'S4':>6} {'pred(len)':>9} {'stance_resid':>12}")
    for resid, name, w, s4, pred in rows_fmt:
        print(f"{name:>15} {w:>7.1f} {s4:>6.2f} {pred:>9.2f} {resid:>+12.2f}")
    json.dump(out, open(f"out/pilot/surface_{TAG}.json", "w"), indent=2)
    print(f"\nwrote out/pilot/surface_{TAG}.json")

if __name__ == "__main__":
    main()
