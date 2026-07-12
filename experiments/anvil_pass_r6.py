# Round-6 (d)/(f) Anvil CPU pass -- no model calls, reads the caches.
#  PART A (f): reconstruct the M=8 budget-sweep per-fact verdict rows from the Qwen
#    cache (the n=100 sweep predates the row-writing version of budget_sweep.py).
#    Uses the chainmetrics/regrade key scheme; BIT-CHECK: recomputed S(k) must match
#    the committed budget_sweep JSON S values exactly, else abort.
#  PART B (d/N5): dial-1 length-mediation of the COLLAPSE arms cross-family --
#    extends crossfam_surface.py (which validated the preservation bonus) to the
#    collapse/neutral arms now present in the 8-arm framing_rows: are node/rolecont/
#    persona ON each family's own length curve (|resid| small), as dial-1 claims?
import hashlib, json
from statistics import mean
from facts import make_facts
from grade import grade
from relay import item_id_for, TEMPLATE_VERSION
from run import FILLER


def part_a():
    tag = "Qwen2.5-7B-Instruct"
    name = f"hf/{tag}"
    cache = json.load(open(f"data/cache_pilot_{tag}.json"))
    facts = make_facts(100, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    ref = {b["budget"]: b["S"] for b in
           json.load(open(f"out/pilot/budget_sweep_{tag}.json"))["budgets"]}
    ks = [0, 1, 2, 4, 8, 16]
    out = open(f"out/pilot/budget_rows_reconstructed_{tag}.jsonl", "w")
    nmiss = 0
    for b in sorted(ref):
        S_chk = {k: [] for k in ks}
        for f in facts:
            pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
            distractors = tuple(pool[:7])
            dctx = ",".join(d.fact_id for d in distractors)
            ctx = hashlib.sha1(f"{FILLER}|{TEMPLATE_VERSION}|{dctx}".encode()).hexdigest()[:8]
            for k in ks:
                iid = item_id_for(f, k, "handoff", b, "light", 0)
                resp = cache.get(f"{name}:{iid}:{ctx}:q")
                if resp is None:
                    nmiss += 1
                    continue
                # bit-check against the COMMITTED numbers uses the LEGACY grader
                # (boundary=False): the committed sweep predates R1.2, and the
                # boundary grader flips a known 0.377% of verdicts downward.
                S_chk[k].append(grade(f, resp, "actionable", boundary=False))
                v = grade(f, resp, "actionable")   # rows use the CURRENT grader
                out.write(json.dumps({"condition": "handoff", "k": k, "fact_id": f.fact_id,
                                      "ftype": f.ftype, "budget": b, "m_facts": 8,
                                      "correct": v}) + "\n")
        got = [round(mean(S_chk[k]), 2) if S_chk[k] else None for k in ks]
        want = [round(x, 2) for x in ref[b]]
        status = "MATCH" if got == want else "MISMATCH"
        print(f"budget {b}: reconstructed(legacy-grader) S={got} vs committed {want} -> {status}")
        assert status == "MATCH", f"bit-check failed at budget {b} -- key scheme wrong, aborting"
    out.close()
    print(f"PART A done, misses={nmiss}")


def part_b():
    import crossfam_surface as cs
    cs.FRAMING_CONDS = ["node", "rolecont", "persona", "distrust", "selfsumm",
                        "neutral_long", "faithful", "handoff"]
    import io, sys
    out = {}
    for m in ["mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-3-medium-4k-instruct"]:
        r = cs.run_family(m)
        out[m.split("/")[-1]] = r
        print(f"=== {m} === curve tau={r['curve']['c']:.4f}*w^{r['curve']['slope']:.2f} "
              f"floor={r['curve']['floor']:.2f}")
        for a in r["arms"]:
            print(f"  {a['arm']:>13} w={a['realized_w']:5} S4_obs={a['S4_obs']} "
                  f"pred(len)={a['S4_pred_from_length']} resid={a['stance_residual']:+.3f}")
    json.dump(out, open("out/pilot/n5_dial1_xfam.json", "w"), indent=2)
    print("PART B done -> out/pilot/n5_dial1_xfam.json")


if __name__ == "__main__":
    part_a()
    part_b()
