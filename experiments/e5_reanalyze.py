# E5 reanalysis from cache (NO API calls): reconstruct the per-fact matrix for all 6
# policies and compute paired-over-facts bootstrap CIs for the FULL set of contrasts
# on BOTH estimands (strict-min = worst-type worst-hop, the pre-registered gate; and
# mean = mean-over-types worst-hop). The pre-registered PRIMARY (dropwatch_stance -
# pin_fair, strict-min) FAILED; these additional contrasts are EXPLORATORY, clearly
# labelled, reported alongside -- not swapped in as the gate.
import hashlib, json, sys
import numpy as np
from facts import make_facts
from grade import grade, _wb
from relay import _planted, _agent_prompt, _truncate, _LOAD_REPS
from run import FILLER
from e5_stance import select_and_stance, compress, POLICIES
from e4dropwatch import WatchState
import random

M, B_ri, K, budget, theta = 24, 4, 16, 25, 0.35
PROBE_KS = [4, 8, 16]


class CacheOnly:
    def __init__(self, name, cache):
        self.name = name; self.cache = cache; self.misses = 0
    def generate(self, prompt, item_id, max_tokens):
        key = f"{self.name}:{item_id}"
        if key not in self.cache:
            self.misses += 1
            return ""            # missing -> empty (should not happen on a complete run)
        return self.cache[key]


def build_mat(tag, name):
    cache = json.load(open(f"data/cache_e5_{tag}.json", encoding="utf-8"))
    be = CacheOnly(name, cache)
    facts = make_facts(100, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    types_cycle = sorted(by_type)
    work = " " + FILLER * _LOAD_REPS["light"]

    def mixed_distractors(probe):
        out = []; pools = {t: [g for g in by_type[t] if g.fact_id != probe.fact_id] for t in types_cycle}
        ptr = {t: 0 for t in types_cycle}; ti = 0
        while len(out) < M - 1:
            t = types_cycle[ti % len(types_cycle)]
            if ptr[t] < len(pools[t]):
                out.append(pools[t][ptr[t]]); ptr[t] += 1
            ti += 1
            if all(ptr[t] >= len(pools[t]) for t in types_cycle):
                break
        return tuple(out[: M - 1])

    def gen(prompt, label, mt):
        h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        return be.generate(prompt, item_id=f"e5:{label}:{h}", max_tokens=mt)

    mat = {p: {t: {} for t in types_cycle} for p in POLICIES}
    for fi, f in enumerate(facts):
        distractors = mixed_distractors(f)
        chain = [f] + list(distractors)
        stmts = [g.statement for g in chain]
        answers = [_wb(g.answer) for g in chain]
        watchable = [g.ftype != "negation" for g in chain]
        for policy in POLICIES:
            rng = random.Random(f"{policy}:{fi}")
            state = {"watch": WatchState(M, B_ri, esc=True),
                     "pin_set": sorted(rng.sample(range(M), B_ri))}
            absent = set(); carry = _planted(f, distractors, seed=0); row = {}
            for h in range(1, K + 1):
                sel, stance = select_and_stance(policy, state, absent, h, M, B_ri, rng)
                if sel:
                    carry = " ".join(stmts[i] for i in sel) + " " + carry
                p = compress(carry, work, stance, h - 1, K, budget)
                msg = gen(p, f"{policy}:{f.fact_id}:h{h}", budget * 3 + 16)
                carry = _truncate(msg, budget)
                wc = _wb(carry)
                absent = {j for j in range(M) if watchable[j] and answers[j] not in wc}
                if h in PROBE_KS:
                    q = _agent_prompt(carry, "", f.query)
                    resp = gen(q, f"{policy}:{f.fact_id}:q{h}", 64)
                    row[h] = grade(f, resp, "actionable")
            mat[policy][f.ftype][f.fact_id] = row
    return mat, types_cycle, be.misses


def estimand(m, types_cycle, idx=None, kind="strictmin"):
    per_type = {}
    for t in types_cycle:
        fids = sorted(m[t])
        sel = fids if idx is None else [fids[i] for i in idx[t]]
        if not sel:
            continue
        per_hop = [np.mean([m[t][fid][k] for fid in sel]) for k in PROBE_KS]
        per_type[t] = float(min(per_hop))     # worst hop for this type
    vals = list(per_type.values())
    return min(vals) if kind == "strictmin" else float(np.mean(vals))


def main(tag="gpt-5.5", name="openai/gpt-5.5"):
    mat, types_cycle, misses = build_mat(tag, name)
    print(f"cache misses: {misses} (should be 0)")
    rng = np.random.default_rng(0)
    sizes = {t: len(mat[POLICIES[0]][t]) for t in types_cycle}
    contrasts = [("dropwatch_stance", "pin_fair", "PRE-REG PRIMARY"),
                 ("pin_fair_faithful", "pin_fair", "exploratory: simple rescue vs baseline"),
                 ("dropwatch_stance", "dropwatch_esc", "stance lever (same re-inject)"),
                 ("pin_fair_faithful", "faithful_always", "pin adds to stance?"),
                 ("dropwatch_stance", "pin_fair_faithful", "adaptive vs simple (does detector help?)")]
    for kind in ("strictmin", "mean"):
        print(f"\n===== estimand = {kind} =====")
        for a, b, label in contrasts:
            diffs = []
            for _ in range(2000):
                idx = {t: rng.integers(0, sizes[t], size=sizes[t]) for t in types_cycle if sizes[t]}
                diffs.append(estimand(mat[a], types_cycle, idx, kind)
                             - estimand(mat[b], types_cycle, idx, kind))
            pt = float(np.mean(diffs))
            lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
            sig = "CI-excl-0" if lo > 0 else "NS"
            print(f"  {a} - {b:>17}  {pt:+.3f} [{lo:+.3f},{hi:+.3f}]  {sig:>9}  ({label})")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
