# SLOT DEPTH SWEEP (prereg out/pilot/slot_depth_predictions.json). e8 (K=6) showed a
# CEILING: slots_copy == slots_regen == 1.00 on numeric/entity -> no headroom to tell
# TRANSCRIPTION (copy) from mere STRUCTURE (regen rewrites the slot each hop). Per arXiv
# 2601.03640, regeneration drift GROWS with the number of regeneration events. So push
# depth: if copy is the real lever, at deep K the REWRITTEN slots (regen) should drift
# down while the COPIED slots (copy) hold -> the copy-minus-regen gap OPENS with K. If
# they still tie at deep K, STRUCTURE alone is the whole effect and copy is not the lever.
import json, os, sys
from statistics import mean
from facts import make_facts, FACT_TYPES
from grade import grade
from run import _build_backend
from e8_structured_slots import run_slots, NDIST, budget

PROBES = int(os.environ.get("SLOT_PROBES", "10"))
KS = [int(x) for x in os.environ.get("SLOT_KS", "8,20").split(",")]


def main(provider="anthropic", model="claude-sonnet-4-6"):
    import e8_structured_slots as e8
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_slots_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)

    facts = make_facts(400, seed=1)
    bt = {}
    for f in facts:
        bt.setdefault(f.ftype, []).append(f)
    res = {"model": model, "budget": budget, "n_probes": PROBES, "ndist": NDIST, "KS": KS,
           "by_k": {}}
    for K in KS:
        e8.K = K  # run_slots reads module-global K
        per_type = {}
        for t in FACT_TYPES:
            probes = bt[t][:PROBES]
            pool = [f for tt in FACT_TYPES for f in bt[tt][PROBES:PROBES + 3]]
            sco, srg, vco, vrg = [], [], [], []
            for probe in probes:
                dist = tuple(d for d in pool if d.fact_id != probe.fact_id)[:NDIST]
                sc, vc = run_slots(backend, probe, dist, 0, "copy")
                sr, vr = run_slots(backend, probe, dist, 0, "regen")
                sco.append(sc); vco.append(vc); srg.append(sr); vrg.append(vr)
            per_type[t] = {"slots_copy": round(mean(sco), 3), "slots_regen": round(mean(srg), 3),
                           "copy_minus_regen": round(mean(sco) - mean(srg), 3),
                           "verbatim_copy": round(mean(vco), 3), "verbatim_regen": round(mean(vrg), 3)}
            with open(f"out/pilot/slot_depth_progress_{tag}.txt", "w") as fh:
                fh.write(f"K={K} done type {t}\n")
        avg_gap = round(mean(per_type[t]["copy_minus_regen"] for t in FACT_TYPES), 3)
        num_gap = per_type["numeric"]["copy_minus_regen"]
        res["by_k"][str(K)] = {"per_type": per_type, "avg_copy_minus_regen": avg_gap,
                               "numeric_copy_minus_regen": num_gap}
    json.dump(res, open(f"out/pilot/e8b_slot_depth_{tag}.json", "w"), indent=2)

    print(f"=== SLOT DEPTH SWEEP ({tag}, ndist={NDIST}, probes/type={PROBES}) ===")
    for K in KS:
        d = res["by_k"][str(K)]
        print(f"\n  K={K}:  {'type':>11} | copy  regen  gap | vbtm_copy vbtm_regen")
        for t in FACT_TYPES:
            c = d["per_type"][t]
            print(f"          {t:>11} | {c['slots_copy']:.2f}  {c['slots_regen']:.2f}  "
                  f"{c['copy_minus_regen']:+.2f} |   {c['verbatim_copy']:.2f}      {c['verbatim_regen']:.2f}")
        print(f"          {'AVG gap':>11} | copy-regen avg {d['avg_copy_minus_regen']:+.3f}  "
              f"(numeric {d['numeric_copy_minus_regen']:+.2f})")
    g8 = res["by_k"][str(KS[0])]["avg_copy_minus_regen"]
    gd = res["by_k"][str(KS[-1])]["avg_copy_minus_regen"]
    print(f"\nGAP OPENS WITH DEPTH? K={KS[0]} gap {g8:+.3f} -> K={KS[-1]} gap {gd:+.3f}  "
          f"(delta {gd - g8:+.3f}); transcription is the lever IFF gap grows with K")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
