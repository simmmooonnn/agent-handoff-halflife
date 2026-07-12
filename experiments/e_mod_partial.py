# Partial preview of the semantic-modulation run from the gpt-5.5 cache ONLY (no API).
# Reconstructs results over the facts fully present in cache_mod_gpt-5.5.json (the run
# died at ~29/75 on OpenAI quota exhaustion). UNDER-POWERED preview, not the final result.
import hashlib, json, sys
import numpy as np
from facts import make_facts
from grade import grade, _wb
from relay import _planted, _agent_prompt, _compress_prompt, _truncate, _LOAD_REPS
from run import FILLER
from semantic_modulation import transcode, M, B_ri, K, budget, PROBE_KS, TYPES, ARMS


def main(tag="gpt-5.5", name="openai/gpt-5.5"):
    cache = json.load(open(f"data/cache_mod_{tag}.json", encoding="utf-8"))
    facts = make_facts(100, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    types_cycle = sorted(by_type)
    work = " " + FILLER * _LOAD_REPS["light"]

    def mixed_distractors(probe):
        out = []
        pools = {t: [g for g in by_type[t] if g.fact_id != probe.fact_id] for t in types_cycle}
        ptr = {t: 0 for t in types_cycle}; ti = 0
        while len(out) < M - 1:
            t = types_cycle[ti % len(types_cycle)]
            if ptr[t] < len(pools[t]):
                out.append(pools[t][ptr[t]]); ptr[t] += 1
            ti += 1
            if all(ptr[t] >= len(pools[t]) for t in types_cycle):
                break
        return tuple(out[: M - 1])

    def get(prompt, label):
        h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        return cache.get(f"{name}:mod:{label}:{h}")

    surv = {a: {t: {} for t in TYPES} for a in ARMS}
    probe_facts = [f for f in facts if f.ftype in TYPES]
    complete = {t: 0 for t in TYPES}
    for probe in probe_facts:
        distractors = mixed_distractors(probe)
        chain = [probe] + list(distractors)
        ok_all = True
        arm_srow = {}
        for arm in ARMS:
            inj = transcode(probe, arm)
            pinned_texts = [inj] + [chain[i].statement for i in range(1, B_ri)]
            carry = _planted(probe, distractors, seed=0)
            srow = {}
            for h in range(1, K + 1):
                carry = " ".join(pinned_texts) + " " + carry
                p = (_compress_prompt(carry, work, budget, "faithful", h - 1, K)
                     if arm == "bare_faithful"
                     else _agent_prompt(carry, work, None, budget=budget))
                msg = get(p, f"{arm}:{probe.fact_id}:h{h}")
                if msg is None:
                    ok_all = False; break
                carry = _truncate(msg, budget)
                if h in PROBE_KS:
                    srow[h] = 1 if _wb(probe.answer) in _wb(carry) else 0
            if not ok_all:
                break
            arm_srow[arm] = srow
        if ok_all:
            for arm in ARMS:
                surv[arm][probe.ftype][probe.fact_id] = arm_srow[arm]
            complete[probe.ftype] += 1

    def worst_hop(m, t):
        fids = sorted(m[t])
        if not fids:
            return None
        return min(np.mean([m[t][fid][k] for fid in fids]) for k in PROBE_KS)

    print(f"=== PARTIAL PREVIEW ({tag}, cache-only) complete facts/type: {complete} ===")
    print("numeric worst-hop carry-survival by arm:")
    for a in ARMS:
        vals = {t: worst_hop(surv[a], t) for t in TYPES}
        nv = vals["numeric"]
        print(f"  {a:>14}: numeric={nv if nv is None else round(nv,3)}  by_type="
              f"{ {t:(None if v is None else round(v,3)) for t,v in vals.items()} }")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
