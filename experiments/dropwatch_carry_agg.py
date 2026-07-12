# Aggregate mechanistic check across ALL numeric probes under pin_oracle (probe is
# always pinned, index 0). For gpt-5.4 vs gpt-5.5, at each probe hop, measure TWO
# things separately: (1) is the answer string present in the final carry text?
# (2) does the model actually ANSWER correctly when queried? A gap between them
# tells us whether facts are DROPPED from the carry vs PRESENT-BUT-UNANSWERED.
import json, hashlib
from statistics import mean
from facts import make_facts
from grade import grade, _wb
from relay import _planted, _agent_prompt, _truncate, _LOAD_REPS
from run import FILLER

M, B_ri, K, budget = 24, 4, 16, 25
PROBE_KS = [4, 8, 16]


def by_type(facts):
    d = {}
    for f in facts:
        d.setdefault(f.ftype, []).append(f)
    return d


def mixed_distractors(f, bt, types_cycle):
    out, pools = [], {t: [g for g in bt[t] if g.fact_id != f.fact_id] for t in types_cycle}
    ptr = {t: 0 for t in types_cycle}; ti = 0
    while len(out) < M - 1:
        t = types_cycle[ti % len(types_cycle)]
        if ptr[t] < len(pools[t]):
            out.append(pools[t][ptr[t]]); ptr[t] += 1
        ti += 1
        if all(ptr[t] >= len(pools[t]) for t in types_cycle):
            break
    return tuple(out[: M - 1])


def run(tag, name):
    cache = json.load(open(f"data/cache_e4_{tag}.json", encoding="utf-8"))
    facts = make_facts(100, seed=0)
    bt = by_type(facts); types_cycle = sorted(bt)
    work = " " + FILLER * _LOAD_REPS["light"]
    present = {k: [] for k in PROBE_KS}       # answer string in carry?
    answered = {k: [] for k in PROBE_KS}      # model answers correctly?
    for probe in bt["numeric"]:
        distractors = mixed_distractors(probe, bt, types_cycle)
        chain = [probe] + list(distractors)
        stmts = [g.statement for g in chain]
        carry = _planted(probe, distractors, seed=0)
        ok = True
        for h in range(1, K + 1):
            carry = " ".join(stmts[i] for i in range(B_ri)) + " " + carry
            p = _agent_prompt(carry, work, None, budget=budget)
            gk = f"{name}:e4:pin_oracle:{probe.fact_id}:h{h}:{hashlib.sha1(p.encode()).hexdigest()[:12]}"
            if gk not in cache:
                ok = False; break
            carry = _truncate(cache[gk], budget)
            if h in PROBE_KS:
                q = _agent_prompt(carry, "", probe.query)
                qk = f"{name}:e4:pin_oracle:{probe.fact_id}:q{h}:{hashlib.sha1(q.encode()).hexdigest()[:12]}"
                if qk not in cache:
                    ok = False; break
                present[h].append(1 if _wb(probe.answer) in _wb(carry) else 0)
                answered[h].append(1 if grade(probe, cache[qk], "actionable") else 0)
        if not ok:
            continue
    print(f"===== {tag} (numeric probes, pin_oracle) =====")
    for k in PROBE_KS:
        if present[k]:
            print(f"  hop {k:>2}: answer-in-carry {mean(present[k]):.2f}   "
                  f"answered-correctly {mean(answered[k]):.2f}   (n={len(present[k])})")
    print()


def main():
    for tag, name in [("gpt-5.4", "openai/gpt-5.4"), ("gpt-5.5", "openai/gpt-5.5")]:
        run(tag, name)


if __name__ == "__main__":
    main()
