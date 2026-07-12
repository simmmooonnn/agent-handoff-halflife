# Mechanistic evidence for the retention axis: replay pin_oracle (which re-injects
# facts {0..B_ri-1} EVERY hop; index 0 = the numeric probe) for one numeric probe on
# gpt-5.4 vs gpt-5.5, reading carry text straight from the DROPWATCH caches (no API
# calls). Shows, hop by hop, whether the just-re-stated number survives the next
# compression. Hypothesis: gpt-5.4 keeps it (repairs stick), gpt-5.5 discards it even
# though it was handed the value that very hop (aggressive compression defeats repair).
import json, hashlib, random
from facts import make_facts
from relay import _planted, _agent_prompt, _truncate, _LOAD_REPS
from run import FILLER

M, B_ri, K, budget = 24, 4, 16, 25


def load_cache(tag):
    return json.load(open(f"data/cache_e4_{tag}.json", encoding="utf-8"))


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


def replay(tag, name, probe, distractors):
    """Replay pin_oracle for one probe from cache; return list of (hop, answer_in_carry, carry)."""
    cache = load_cache(tag)
    work = " " + FILLER * _LOAD_REPS["light"]
    chain = [probe] + list(distractors)
    stmts = [g.statement for g in chain]
    ans = probe.answer
    carry = _planted(probe, distractors, seed=0)
    trace = []
    for h in range(1, K + 1):
        sel = list(range(B_ri))                       # pin_oracle: facts 0..B_ri-1 every hop
        carry = " ".join(stmts[i] for i in sel) + " " + carry
        p = _agent_prompt(carry, work, None, budget=budget)
        key = f"{name}:e4:pin_oracle:{probe.fact_id}:h{h}:{hashlib.sha1(p.encode()).hexdigest()[:12]}"
        if key not in cache:
            trace.append((h, None, "<not cached>")); break
        msg = cache[key]
        carry = _truncate(msg, budget)
        trace.append((h, ans in carry, carry))
    return trace


def main():
    facts = make_facts(100, seed=0)
    bt = by_type(facts)
    types_cycle = sorted(bt)
    # first numeric probe
    probe = next(f for f in facts if f.ftype == "numeric")
    distractors = mixed_distractors(probe, bt, types_cycle)
    print(f"PROBE: {probe.statement!r}  answer={probe.answer!r}")
    print(f"(pin_oracle re-injects this statement at the FRONT of the carry every hop)\n")
    for tag, name in [("gpt-5.4", "openai/gpt-5.4"), ("gpt-5.5", "openai/gpt-5.5")]:
        trace = replay(tag, name, probe, distractors)
        kept = sum(1 for _, a, _ in trace if a)
        print(f"===== {tag}: number present in carry {kept}/{len(trace)} hops =====")
        for h, a, carry in trace:
            mark = "KEEP" if a else ("DROP" if a is False else "????")
            print(f"  h{h:>2} [{mark}] {carry[:160]}")
        print()


if __name__ == "__main__":
    main()
