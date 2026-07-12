# E5 -- DROPWATCH-S: stance-escalating closed-loop controller for AGGRESSIVE
# compressors (the models where the original pin-based DROPWATCH failed, e.g. gpt-5.5).
#
# WHY (2026-07-07 rescue diagnostic): on gpt-5.5 the pin_oracle re-injection ceiling
# for numeric facts is ~0.04-0.20 under NEUTRAL compression, but rises to 0.52 [.32,.72]
# when the per-hop compression uses a PRESERVATION stance. The lever that rescues the
# failure is the compressor's STANCE (what it is told to DO), not which facts are
# re-injected and not content salience-labeling (salience arm failed). So the original
# DROPWATCH (adaptive re-injection under neutral compression) cannot win on aggressive
# compressors, but adding stance-escalation to the same detect+repair loop should.
#
# Six policies isolate each lever (regime identical to E4: M=24 mixed, K=16, budget=25,
# B_ri=4, theta=0.35). sel = facts re-injected at the front this hop; stance = neutral
# or faithful compression this hop.
#   pin_fair          sel=fixed random B_ri, stance=neutral    [original E4 gate baseline]
#   dropwatch_esc     sel=adaptive+escalate,  stance=neutral    [original method; fails on gpt-5.5]
#   faithful_always   sel=[] (no re-inject),  stance=faithful   [pure Moat-2 baseline: "just always preserve"]
#   pin_fair_faithful sel=fixed random B_ri, stance=faithful    [fixed pin + stance; isolates adaptivity]
#   dropwatch_stance  sel=adaptive+escalate,  stance=faithful WHEN repairing, neutral when quiescent  [THE METHOD]
#   pin_oracle        sel=range(B_ri),        stance=neutral    [upper anchor, continuity]
#
# Pre-registered comparisons (out/pilot/e5_stance_predictions.json, committed before run):
#   PRIMARY (rescue gate): dropwatch_stance - pin_fair, CI-excl-0, cost <= pin_fair.
#   STANCE lever:          dropwatch_stance - dropwatch_esc, CI-excl-0.
#   ADAPTIVITY control:    dropwatch_stance - faithful_always (does the detector earn its keep?).
import hashlib, json, os, random, sys
import numpy as np
from facts import make_facts
from grade import grade, _wb
from relay import _planted, _agent_prompt, _compress_prompt, _truncate, _LOAD_REPS
from run import FILLER, _build_backend
from e4dropwatch import WatchState

PROBE_KS = [4, 8, 16]
POLICIES = ["pin_fair", "dropwatch_esc", "faithful_always",
            "pin_fair_faithful", "dropwatch_stance", "pin_oracle"]
GATE_BASELINE = "pin_fair"
METHOD = "dropwatch_stance"


def select_and_stance(policy, state, absent, hop, M, B_ri, rng):
    """Return (sel list of fact indices to re-inject, stance str)."""
    if policy == "pin_fair":
        return state["pin_set"], "neutral"
    if policy == "pin_oracle":
        return list(range(B_ri)), "neutral"
    if policy == "faithful_always":
        return [], "faithful"
    if policy == "pin_fair_faithful":
        return state["pin_set"], "faithful"
    if policy == "dropwatch_esc":
        return state["watch"].select(absent, hop), "neutral"
    # dropwatch_stance: adaptive repair + faithful compression on any repairing hop
    sel = state["watch"].select(absent, hop)
    return sel, ("faithful" if sel else "neutral")


def compress(carry, work, stance, hop, K, budget):
    if stance == "neutral":
        return _agent_prompt(carry, work, None, budget=budget)
    return _compress_prompt(carry, work, budget, "faithful", hop, K)


def main(provider="openai", model="gpt-5.5", n="100", K="16",
         budget="25", m_facts="24", b_ri="4", theta="0.35"):
    n = int(n); K = int(K); budget = int(budget); M = int(m_facts)
    B_ri = int(b_ri); theta = float(theta)
    probe_ks = [k for k in PROBE_KS if k <= K]
    facts = make_facts(n, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    types_cycle = sorted(by_type)
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_e5_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)
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

    def gen(prompt, label, max_tokens):
        h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        return backend.generate(prompt, item_id=f"e5:{label}:{h}", max_tokens=max_tokens)

    mat = {p: {t: {} for t in types_cycle} for p in POLICIES}
    cost = {p: [] for p in POLICIES}          # executed re-injection slots per hop
    fhops = {p: [] for p in POLICIES}         # fraction of hops compressed under faithful
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
            absent = set()
            carry = _planted(f, distractors, seed=0)
            row, ncalls, nfaith = {}, 0, 0
            for h in range(1, K + 1):
                sel, stance = select_and_stance(policy, state, absent, h, M, B_ri, rng)
                ncalls += len(sel)
                nfaith += (stance == "faithful")
                if sel:
                    carry = " ".join(stmts[i] for i in sel) + " " + carry
                p = compress(carry, work, stance, h - 1, K, budget)
                msg = gen(p, f"{policy}:{f.fact_id}:h{h}", budget * 3 + 16)
                carry = _truncate(msg, budget)
                wc = _wb(carry)
                absent = {j for j in range(M) if watchable[j] and answers[j] not in wc}
                if h in probe_ks:
                    q = _agent_prompt(carry, "", f.query)
                    resp = gen(q, f"{policy}:{f.fact_id}:q{h}", 64)
                    row[h] = grade(f, resp, "actionable")
            mat[policy][f.ftype][f.fact_id] = row
            cost[policy].append(ncalls / K)
            fhops[policy].append(nfaith / K)
        with open(f"out/pilot/e5_progress_{tag}.txt", "w") as fh:
            fh.write(f"done fact {fi + 1}/{len(facts)}\n")

    def strictmin(m, idx=None):
        per_type = {}
        for t in types_cycle:
            fids = sorted(m[t])
            sel = fids if idx is None else [fids[i] for i in idx[t]]
            if not sel:
                continue
            per_hop = [np.mean([m[t][fid][k] for fid in sel]) for k in probe_ks]
            per_type[t] = float(min(per_hop))
        return (min(per_type.values()), float(np.mean(list(per_type.values()))), per_type)

    rng = np.random.default_rng(0)
    sizes = {t: len(mat[POLICIES[0]][t]) for t in types_cycle}
    boot = {p: [] for p in POLICIES}
    diffs = {"stance_minus_pinfair": [], "stance_minus_dwesc": [], "stance_minus_faithalways": []}
    for _ in range(2000):
        idx = {t: rng.integers(0, sizes[t], size=sizes[t]) for t in types_cycle if sizes[t]}
        vals = {p: strictmin(mat[p], idx)[0] for p in POLICIES}
        for p in POLICIES:
            boot[p].append(vals[p])
        diffs["stance_minus_pinfair"].append(vals[METHOD] - vals["pin_fair"])
        diffs["stance_minus_dwesc"].append(vals[METHOD] - vals["dropwatch_esc"])
        diffs["stance_minus_faithalways"].append(vals[METHOD] - vals["faithful_always"])
    res = {"model": model, "n": n, "K": K, "M": M, "B_ri": B_ri, "theta": theta,
           "budget": budget, "probe_ks": probe_ks, "arms": {}}
    for p in POLICIES:
        mn, me, pt = strictmin(mat[p])
        res["arms"][p] = {"strict_min": round(mn, 3), "mean": round(me, 3),
                          "min_ci": [round(float(np.percentile(boot[p], 2.5)), 3),
                                     round(float(np.percentile(boot[p], 97.5)), 3)],
                          "cost_per_hop": round(float(np.mean(cost[p])), 2),
                          "faithful_hop_frac": round(float(np.mean(fhops[p])), 2),
                          "per_type": {t: round(v, 3) for t, v in pt.items()}}

    def ci(key):
        return [round(float(np.percentile(diffs[key], 2.5)), 3),
                round(float(np.percentile(diffs[key], 97.5)), 3)],\
               round(float(np.mean(diffs[key])), 3)

    pf_ci, pf_pt = ci("stance_minus_pinfair")
    dw_ci, dw_pt = ci("stance_minus_dwesc")
    fa_ci, fa_pt = ci("stance_minus_faithalways")
    gate = pf_ci[0] > 0 and res["arms"][METHOD]["cost_per_hop"] <= res["arms"]["pin_fair"]["cost_per_hop"] + 1e-9
    res["comparisons"] = {
        "PRIMARY_stance_minus_pinfair": {"point": pf_pt, "ci": pf_ci},
        "stance_minus_dwesc": {"point": dw_pt, "ci": dw_ci},
        "stance_minus_faithalways": {"point": fa_pt, "ci": fa_ci}}
    res["gate_pass"] = bool(gate)
    json.dump(res, open(f"out/pilot/e5_stance_{tag}.json", "w"), indent=2)
    print(f"=== E5 DROPWATCH-S ({tag}, M={M}, B_ri={B_ri}, K={K}) ===")
    for p in POLICIES:
        r = res["arms"][p]
        print(f"  {p:>17} strict-min={r['strict_min']:.3f} [{r['min_ci'][0]:.3f},{r['min_ci'][1]:.3f}] "
              f"mean={r['mean']:.3f} cost={r['cost_per_hop']} faith_hops={r['faithful_hop_frac']}")
    print(f"\nPRIMARY  dropwatch_stance - pin_fair:      {pf_pt:+.3f} {pf_ci}")
    print(f"stance lever  stance - dropwatch_esc:       {dw_pt:+.3f} {dw_ci}")
    print(f"adaptivity    stance - faithful_always:     {fa_pt:+.3f} {fa_ci}")
    print("GATE:", "PASS -> stance-escalation rescues the method on an aggressive compressor"
          if gate else "FAIL -> report honestly")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
