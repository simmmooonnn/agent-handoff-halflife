# SCARCITY SHOWDOWN (prereg out/pilot/scarcity_showdown_predictions.json). e9 showed
# Mem0-style memory holds a SINGLE number -> the honest win must be under SCARCE budget +
# MANY competing fields, where every method can only keep a few facts and WHICH ones is
# the question. Head-to-head at an IDENTICAL per-hop carry budget B:
#   free_text = compress all M facts into B words of prose (no protection)   [naive]
#   mem0      = maintain a B-word extracted fact-list memory (Mem0 salience) [SOTA baseline]
#   ours      = split B into verbatim [SLOTS] for the P measured-most-fragile facts
#               (budget-exempt) + a compressed [NOTES] section for the rest   [OURS]
# Allocation for ours uses LEAVE-ONE-CHAIN-OUT per-type free_text survival (measured, no
# peeking). Metric: TOTAL survival over all M facts at depth K. The claim: measurement
# beats Mem0's fragility-blind salience heuristic when the worst type is non-obvious.
import hashlib, json, os, sys
from statistics import mean
import numpy as np
from facts import make_facts, FACT_TYPES
from grade import grade
from relay import _agent_prompt, _truncate, _LOAD_REPS
from run import FILLER, _build_backend
from e9_sota_showdown import _mem0_extract_prompt, _mem0_merge_prompt

M = int(os.environ.get("SC_M", "12"))
K = int(os.environ.get("SC_K", "4"))
C = int(os.environ.get("SC_C", "10"))
budget = int(os.environ.get("SC_BUDGET", "25"))
P = int(os.environ.get("SC_P", "4"))           # protection budget (facts ours slots)
PER_TYPE = M // 4
LOAD = "light"
RNG = np.random.default_rng(0)


def build_chains():
    facts = make_facts(400, seed=1)
    bt = {}
    for f in facts:
        bt.setdefault(f.ftype, []).append(f)
    chains, ptr = [], {t: 0 for t in FACT_TYPES}
    for c in range(C):
        chain = []
        for t in FACT_TYPES:
            for _ in range(PER_TYPE):
                chain.append(bt[t][ptr[t]]); ptr[t] += 1
        chains.append(chain)
    return chains


def _plant(chain, seed):
    import random
    stmts = [f.statement for f in chain]
    random.Random(f"sc:{seed}").shuffle(stmts)
    return " ".join(stmts)


def _query_all(backend, carry, chain, iid):
    surv = {}
    for f in chain:
        r = backend.generate(_agent_prompt(carry, "", f.query),
                             item_id=f"{iid}:q:{f.fact_id}", max_tokens=64)
        surv[f.fact_id] = (f.ftype, 1 if grade(f, r, "actionable") else 0)
    return surv


def run_free_text(backend, chain, ci):
    work = (" " + FILLER) * _LOAD_REPS[LOAD]
    iid = f"sc_ft:{ci}:{hashlib.sha1(_plant(chain,0).encode()).hexdigest()[:10]}"
    carry = _plant(chain, 0)
    for i in range(K):
        p = _agent_prompt(carry, work, None, budget=budget)
        carry = _truncate(backend.generate(p, item_id=f"{iid}:h{i}", max_tokens=budget * 3 + 16), budget)
    return _query_all(backend, carry, chain, iid)


def run_mem0(backend, chain, ci):
    work = (" " + FILLER) * _LOAD_REPS[LOAD]
    iid = f"sc_m0:{ci}:{hashlib.sha1(_plant(chain,0).encode()).hexdigest()[:10]}"
    mem = _truncate(backend.generate(_mem0_extract_prompt(_plant(chain, 0), budget),
                                     item_id=f"{iid}:e", max_tokens=budget * 3 + 16), budget * 2)
    for i in range(1, K):
        mem = _truncate(backend.generate(_mem0_merge_prompt(mem, work.strip(), budget),
                                         item_id=f"{iid}:m{i}", max_tokens=budget * 3 + 16), budget * 2)
    return _query_all(backend, mem, chain, iid)


def _multislot_carry(slotted, notes):
    import random
    ns = [f.statement for f in notes]
    random.Random("scms").shuffle(ns)
    return "[SLOTS]\n" + "\n".join(f.statement for f in slotted) + "\n[NOTES]\n" + " ".join(ns)


def run_ours(backend, chain, ci, slot_types_order):
    # choose the P facts whose TYPE is most fragile by calibration (slot_types_order: types
    # ascending by measured survival); fill slots type-by-type up to P.
    work = (" " + FILLER) * _LOAD_REPS[LOAD]
    ranked = sorted(chain, key=lambda f: slot_types_order.index(f.ftype))
    slotted = ranked[:P]
    notes = ranked[P:]
    notes_budget = max(4, budget - 4 * P)  # slots eat ~4 words each; notes get the rest
    iid = f"sc_ours:{ci}:P{P}:{hashlib.sha1(_plant(chain,0).encode()).hexdigest()[:10]}"
    carry = _multislot_carry(slotted, notes)
    for i in range(K):
        head = ("Compress this running context. Reproduce the [SLOTS] section VERBATIM "
                "(character for character, do not drop, round, or paraphrase any slot). Then "
                f"compress the [NOTES] section to at most {notes_budget} words. Output:\n"
                "[SLOTS]\n<verbatim>\n[NOTES]\n<compressed>")
        p = head + "\n" + carry + (f"\nWORK: {work}" if work else "")
        carry = _truncate(backend.generate(p, item_id=f"{iid}:h{i}", max_tokens=budget * 5 + 40), budget * 3)
    return _query_all(backend, carry, chain, iid)


def total(surv):
    return mean(s for _, (_, s) in surv.items()) if surv else 0.0


def main(provider="anthropic", model="claude-sonnet-4-6"):
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_scarcity_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)
    chains = build_chains()

    # pass 1: free_text (baseline + calibration source) and mem0 (SOTA)
    ft = [run_free_text(backend, ch, ci) for ci, ch in enumerate(chains)]
    m0 = [run_mem0(backend, ch, ci) for ci, ch in enumerate(chains)]

    def cal_type_order(exclude_ci):
        agg = {}
        for ci, s in enumerate(ft):
            if ci == exclude_ci:
                continue
            for _, (t, v) in s.items():
                agg.setdefault(t, []).append(v)
        surv = {t: mean(v) for t, v in agg.items()}
        return sorted(FACT_TYPES, key=lambda t: surv.get(t, 1.0))  # most fragile first

    # pass 2: ours (measurement-driven multislot), leave-one-chain-out calibration
    ours = [run_ours(backend, ch, ci, cal_type_order(ci)) for ci, ch in enumerate(chains)]

    # allocation references (modeled on free_text per-fact survival, like e7): oracle/folk/random @P
    def alloc_ref(policy):
        vals = []
        for ci, s in enumerate(ft):
            items = list(s.items())  # [(fid,(t,v))]
            if policy == "oracle":
                order = sorted(range(len(items)), key=lambda i: items[i][1][1])
            elif policy == "folk":
                order = sorted(range(len(items)), key=lambda i: (0 if items[i][1][0] == "numeric" else 1))
            else:  # random
                sub = []
                for _ in range(200):
                    pr = set(RNG.choice(len(items), size=P, replace=False).tolist())
                    sub.append(mean(1 if i in pr else items[i][1][1] for i in range(len(items))))
                vals.append(mean(sub)); continue
            pr = set(order[:P])
            vals.append(mean(1 if i in pr else items[i][1][1] for i in range(len(items))))
        return round(mean(vals), 3)

    res = {"model": model, "M": M, "K": K, "C": C, "budget": budget, "P": P,
           "total": {"free_text": round(mean(total(s) for s in ft), 3),
                     "mem0": round(mean(total(s) for s in m0), 3),
                     "ours": round(mean(total(s) for s in ours), 3),
                     "ref_oracle@P": alloc_ref("oracle"), "ref_folk@P": alloc_ref("folk"),
                     "ref_random@P": alloc_ref("random")}}
    # per-type survival (free_text) to expose the fragility ranking mem0 is blind to
    agg = {}
    for s in ft:
        for _, (t, v) in s.items():
            agg.setdefault(t, []).append(v)
    res["free_text_per_type"] = {t: round(mean(v), 3) for t, v in sorted(agg.items())}
    res["ours_minus_mem0"] = round(res["total"]["ours"] - res["total"]["mem0"], 3)
    json.dump(res, open(f"out/pilot/e10_scarcity_showdown_{tag}.json", "w"), indent=2)

    print(f"=== SCARCITY SHOWDOWN ({tag}, M={M}, K={K}, C={C}, budget={budget}, P={P}) ===")
    print(f"free_text per-type survival (the ranking mem0 can't see): {res['free_text_per_type']}")
    print(f"  TOTAL survival over {M} facts:")
    for k, v in res["total"].items():
        print(f"    {k:>14}: {v:.3f}")
    print(f"  OURS - Mem0 = {res['ours_minus_mem0']:+.3f}  (>0 => measurement beats SOTA salience heuristic)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
