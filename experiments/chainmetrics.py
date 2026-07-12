# R1.4: relate our probe-based usable-recall S(k) to FActScore-style CONTENT
# metrics computed directly on the cached chain texts -- answers the round-4 ask
# "compare empirically to telephone/iterated-summarization degradation metrics"
# without any new model calls.
#
# For each chain (fact f, condition, k=KMAX) we reconstruct the per-hop carry
# messages from the cache (keys {name}:{iid}:{ctx}:h{i}) and compute, at every
# hop h, the fraction of the M=8 planted atomic facts (probe + distractors)
# whose gold answers appear word-boundary in the carry text. That is a
# deterministic FActScore analog (atomic-fact precision against the planted
# set). We report:
#   C(h)   content survival (all 8 planted facts)          -- telephone-style curve
#   P(h)   probe-answer text presence                       -- text vs recall gap
#   S(k)   probe usable-recall (the paper's metric, from rows)
# Key comparisons: does C(h) decay exponentially to a floor like S(k)? Is
# text-presence at the final hop necessary/sufficient for usable recall?
#
# Run on Anvil (cache lives there); CPU-only.
import hashlib, json, sys
from statistics import mean
from facts import make_facts
from grade import _wb
from relay import item_id_for, TEMPLATE_VERSION
from run import FILLER

MODEL_TAG = "Qwen2.5-7B-Instruct"
CONDS = ["handoff", "manifest", "ledger", "node"]
KMAX = {"handoff": 16, "manifest": 8, "ledger": 8, "node": 4}
BUDGET, M_FACTS, LOAD, SEED, N = 25, 8, "light", 0, 100


def main():
    cache = json.load(open(f"data/cache_pilot_{MODEL_TAG}.json"))
    name = f"hf/{MODEL_TAG}"
    facts = make_facts(N, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)

    out = {}
    for cond in CONDS:
        kmax = KMAX[cond]
        C = {h: [] for h in range(kmax)}
        P = {h: [] for h in range(kmax)}
        used, missing = 0, 0
        for f in facts:
            pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
            distractors = tuple(pool[:M_FACTS - 1])
            iid = item_id_for(f, kmax, cond, BUDGET, LOAD, SEED)
            dctx = ",".join(d.fact_id for d in distractors)
            ctx = hashlib.sha1(f"{FILLER}|{TEMPLATE_VERSION}|{dctx}".encode()
                               ).hexdigest()[:8]
            answers = [(g.answer, g.ftype) for g in [f] + list(distractors)]
            ok = True
            for h in range(kmax):
                msg = cache.get(f"{name}:{iid}:{ctx}:h{h}")
                if msg is None:
                    ok = False
                    break
                w = _wb(msg)
                # negation answers are yes/no -- text presence is meaningless, skip
                present = [(_wb(a) in w) for a, t in answers if t != "negation"]
                C[h].append(mean(present) if present else 0.0)
                if f.ftype != "negation":
                    P[h].append(float(_wb(f.answer) in w))
            used += ok
            missing += (not ok)
        out[cond] = {
            "k_max": kmax, "chains": used, "missing": missing,
            "content_survival_C": [round(mean(C[h]), 3) for h in range(kmax) if C[h]],
            "probe_presence_P": [round(mean(P[h]), 3) for h in range(kmax) if P[h]],
        }
        print(f"{cond:10} (k={kmax}, {used} chains, {missing} missing)")
        print(f"  C(h) planted-content: {out[cond]['content_survival_C']}")
        print(f"  P(h) probe-presence : {out[cond]['probe_presence_P']}")
    json.dump(out, open(f"out/pilot/chainmetrics_{MODEL_TAG}.json", "w"), indent=1)
    print("wrote out/pilot/chainmetrics_" + MODEL_TAG + ".json")


if __name__ == "__main__":
    main()
