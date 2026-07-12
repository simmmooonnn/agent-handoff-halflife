# Confirmation run for the itemize discovery, with the length confound CONTROLLED.
#
# protocol_ablate found itemize S(4)=0.87 [0.80,0.93] >> faithful 0.58 [0.48,0.68] --
# but protolens found itemize outputs ~42 words vs the 25-word budget (93% exceed it):
# the win is partly a budget increase in disguise (format instruction overrides the
# budget instruction). Fair test = compare at MATCHED REALIZED LENGTH:
#   pair A (load-bearing): itemize@25 (realized ~42) vs faithful@50 (realized ~0.7*50)
#   pair B (reverse):      itemize@12 vs faithful@25 (realized ~22)
# plus k=8 for durability of the flat itemize curve.
#
# PRE-REGISTERED: the FORMAT effect is real iff itemize beats faithful on S(4) CI at
# matched realized length (pair A counts if realized means are within +-25%). If
# itemize@12 still realizes ~40 words (format fully overrides budget), report the
# intervention honestly as a format+length BUNDLE and use pair A as the fair costed
# comparison. SECONDARY: itemize S(8) CI (durability).
import hashlib, json, os, sys
from statistics import mean
import numpy as np
from facts import make_facts
from relay import item_id_for, TEMPLATE_VERSION
from run import run_sweep, FILLER, _build_backend
from analyze import survival_curve, empirical_chance

PAIRS = [("itemize", 25), ("faithful", 50), ("itemize", 12), ("faithful", 25),
         ("handoff", 25)]  # handoff@25 = anchor, fully cached

def boot_S_at_k(rows, label, k, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    sub = [r for r in rows if r["label"] == label and r["k"] == k]
    if not sub:
        return None
    fids = sorted({r["fact_id"] for r in sub})
    by = {f: [r["correct"] for r in sub if r["fact_id"] == f] for f in fids}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(fids, size=len(fids), replace=True)
        v = [c for f in draw for c in by[f]]
        vals.append(sum(v) / len(v))
    a = np.array(vals)
    point = sum(v for f in fids for v in by[f]) / sum(len(by[f]) for f in fids)
    return point, float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def hop_lengths(cache, prefix, facts, by_type, cond, budget, m_facts, ks):
    lens = []
    for f in facts:
        pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
        distractors = tuple(pool[:m_facts - 1])
        dctx = ",".join(d.fact_id for d in distractors)
        ctx = hashlib.sha1(f"{FILLER}|{TEMPLATE_VERSION}|{dctx}".encode()).hexdigest()[:8]
        for k in ks:
            if k == 0:
                continue
            iid = item_id_for(f, k, cond, budget, "light", 0)
            for i in range(k):
                key = f"{prefix}:{iid}:{ctx}:h{i}"
                if key in cache:
                    lens.append(len(cache[key].split()))
    return lens

def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100",
         ks="0,1,2,4,8", m_facts="8"):
    n = int(n); m_facts = int(m_facts)
    ks_list = [int(x) for x in str(ks).split(",")]
    facts = make_facts(n, seed=0)
    tag = model.split('/')[-1]
    backend = _build_backend(provider, model, f"data/cache_pilot_{tag}.json")
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    os.makedirs("out/pilot", exist_ok=True)
    all_rows = []
    for cond, budget in PAIRS:
        rows = run_sweep(backend, facts, ks=ks_list, conditions=[cond], budget=budget,
                         load="light", filler=FILLER, seeds=[0], level="actionable",
                         m_facts=m_facts)
        for r in rows:
            r["label"] = f"{cond}@{budget}"
            r["budget"] = budget
        all_rows.extend(rows)
        with open(f"out/pilot/proto2_rows_{tag}.jsonl", "w") as fh:  # incremental
            for r in all_rows:
                fh.write(json.dumps(r) + "\n")
    # chance from cached nofact at budget=25
    ch_rows = run_sweep(backend, facts, ks=[0], conditions=["nofact"], budget=25,
                        load="light", filler=FILLER, seeds=[0], level="actionable",
                        m_facts=m_facts)
    chance = empirical_chance(ch_rows)["overall"]

    prefix = "hf/" + tag
    cache = backend._cache
    res = {"model": model, "n": n, "m_facts": m_facts, "chance": chance, "arms": {}}
    k_mid = 4 if 4 in ks_list else ks_list[-1]
    k_deep = max(ks_list)
    print(f"=== proto2: matched-realized-length confirmation (n={n}, M={m_facts}, "
          f"chance={chance:.3f}) ===")
    print(f"{'arm':>13} {'S(k)':<32} {'S(4)[CI]':>18} {'S(8)[CI]':>18} {'real_w':>7} {'%>stated':>9}")
    for cond, budget in PAIRS:
        label = f"{cond}@{budget}"
        sub = [r for r in all_rows if r["label"] == label]
        kk = sorted({r["k"] for r in sub})
        S = []
        for k in kk:
            v = [r["correct"] for r in sub if r["k"] == k]
            S.append(sum(v) / len(v))
        b4 = boot_S_at_k(all_rows, label, k_mid)
        b8 = boot_S_at_k(all_rows, label, k_deep)
        lens = hop_lengths(cache, prefix, facts, by_type, cond, budget, m_facts, ks_list)
        mw = mean(lens) if lens else float("nan")
        over = 100.0 * sum(1 for x in lens if x > budget) / len(lens) if lens else float("nan")
        res["arms"][label] = {"S": S, "ks": kk,
                              "S4": b4, "S8": b8, "realized_words": mw, "pct_over": over}
        s4txt = f"{b4[0]:.2f} [{b4[1]:.2f},{b4[2]:.2f}]" if b4 else "-"
        s8txt = f"{b8[0]:.2f} [{b8[1]:.2f},{b8[2]:.2f}]" if b8 else "-"
        print(f"{label:>13} {str([round(x,2) for x in S]):<32} {s4txt:>18} {s8txt:>18} "
              f"{mw:>7.1f} {over:>8.1f}%", flush=True)
    json.dump(res, open(f"out/pilot/proto2_{tag}.json", "w"), indent=2)

    A = res["arms"]
    print("--- pre-registered verdicts ---")
    # pair A: itemize@25 vs faithful@50 at matched realized length
    ia, f50 = A.get("itemize@25"), A.get("faithful@50")
    if ia and f50:
        matched = abs(ia["realized_words"] - f50["realized_words"]) <= 0.25 * max(
            ia["realized_words"], f50["realized_words"])
        sep = ia["S4"][1] > f50["S4"][2]
        print(f"  pair A realized words: itemize@25={ia['realized_words']:.1f} vs "
              f"faithful@50={f50['realized_words']:.1f} -> matched={matched}")
        print(f"  pair A S(4): itemize CI above faithful: {sep}")
        if matched and sep:
            print("  -> FORMAT EFFECT REAL at matched realized length")
        elif matched:
            print("  -> at matched length the format effect does NOT separate: itemize's win was the length")
    ib, f25 = A.get("itemize@12"), A.get("faithful@25")
    if ib and f25:
        print(f"  pair B realized words: itemize@12={ib['realized_words']:.1f} vs "
              f"faithful@25={f25['realized_words']:.1f}; "
              f"S(4) itemize CI above faithful: {ib['S4'][1] > f25['S4'][2]}")
    if ia and ia["S8"]:
        print(f"  durability: itemize@25 S(8) = {ia['S8'][0]:.2f} [{ia['S8'][1]:.2f},{ia['S8'][2]:.2f}]")

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
