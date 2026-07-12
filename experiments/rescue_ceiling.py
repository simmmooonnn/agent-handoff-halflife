# RESCUE CEILING TEST (prereg: out/pilot/rescue_ceiling_predictions.json, commit 1fcc137).
# On gpt-5.5, run a pin_oracle-style chain (re-inject facts 0..B_ri-1 at the FRONT
# every hop) for numeric probes, varying only the per-hop COMPRESSION STANCE / injection
# frame. Measures whether an upgraded repair PRIMITIVE lifts the 0.20 carry-survival
# ceiling that bare re-injection + neutral compression hits. Reads/writes its own cache.
import hashlib, json, os, sys
from statistics import mean
import numpy as np
from facts import make_facts
from grade import grade, _wb
from relay import _planted, _agent_prompt, _compress_prompt, _truncate, _LOAD_REPS
from run import FILLER, _build_backend

M, B_ri, K, budget = 24, 4, 16, 25
PROBE_KS = [4, 8, 16]
STANCES = ["neutral", "faithful", "ledger", "salience"]


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


def inject(stmts_sel, stance):
    # how the re-stated facts are prepended to the carry before compression
    if stance == "salience":
        facts = " ".join(stmts_sel)
        return (f"CRITICAL - the next step is tested on these exact values; preserve "
                f"them verbatim: {facts}")
    return " ".join(stmts_sel)                      # bare declarative (neutral/faithful/ledger)


def compress_prompt(carry, work, stance, hop):
    if stance in ("neutral", "salience"):
        return _agent_prompt(carry, work, None, budget=budget)   # neutral compression
    mode = {"faithful": "faithful", "ledger": "ledger"}[stance]
    return _compress_prompt(carry, work, budget, mode, hop, K)


def main(provider="openai", model="gpt-5.5"):
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_rescue_{tag}.json")
    facts = make_facts(100, seed=0)
    bt = by_type(facts); types_cycle = sorted(bt)
    work = " " + FILLER * _LOAD_REPS["light"]
    os.makedirs("out/pilot", exist_ok=True)

    def gen(prompt, label, mt):
        h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        return backend.generate(prompt, item_id=f"rescue:{label}:{h}", max_tokens=mt)

    surv = {s: {k: [] for k in PROBE_KS} for s in STANCES}     # answer-in-carry
    ans = {s: {k: [] for k in PROBE_KS} for s in STANCES}      # answered correctly
    for probe in bt["numeric"]:
        distractors = mixed_distractors(probe, bt, types_cycle)
        chain = [probe] + list(distractors)
        stmts = [g.statement for g in chain]
        for stance in STANCES:
            carry = _planted(probe, distractors, seed=0)
            for h in range(1, K + 1):
                carry = inject([stmts[i] for i in range(B_ri)], stance) + " " + carry
                p = compress_prompt(carry, work, stance, h - 1)
                msg = gen(p, f"{stance}:{probe.fact_id}:h{h}", budget * 3 + 16)
                carry = _truncate(msg, budget)
                if h in PROBE_KS:
                    surv[stance][h].append(1 if _wb(probe.answer) in _wb(carry) else 0)
                    q = _agent_prompt(carry, "", probe.query)
                    r = gen(q, f"{stance}:{probe.fact_id}:q{h}", 64)
                    ans[stance][h].append(1 if grade(probe, r, "actionable") else 0)

    rng = np.random.default_rng(0)
    n = len(bt["numeric"])
    res = {"model": model, "regime": {"M": M, "B_ri": B_ri, "K": K, "budget": budget, "n_numeric": n},
           "stances": {}}
    base_minhop = None
    for s in STANCES:
        # min over probe hops of mean carry-survival (worst-hop, matches strict-min spirit)
        per_hop = {k: mean(surv[s][k]) for k in PROBE_KS}
        per_hop_ans = {k: mean(ans[s][k]) for k in PROBE_KS}
        minhop = min(per_hop.values())
        # bootstrap CI of min-over-hop carry-survival over facts
        boot = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            boot.append(min(mean([surv[s][k][i] for i in idx]) for k in PROBE_KS))
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        res["stances"][s] = {"carry_survival_by_hop": {k: round(v, 3) for k, v in per_hop.items()},
                             "answered_by_hop": {k: round(v, 3) for k, v in per_hop_ans.items()},
                             "min_hop_carry_survival": round(minhop, 3),
                             "min_hop_ci": [round(x, 3) for x in ci]}
        if s == "neutral":
            base_minhop = minhop
    # verdict vs pre-registered readout
    lifts = {s: res["stances"][s]["min_hop_carry_survival"] for s in STANCES if s != "neutral"}
    best = max(lifts, key=lifts.get)
    best_val = lifts[best]
    best_ci_lo = res["stances"][best]["min_hop_ci"][0]
    if best_val >= 0.40 and best_ci_lo > base_minhop:
        verdict = f"RESCUE_EXISTS: '{best}' lifts ceiling to {best_val:.2f} (CI-lo {best_ci_lo:.2f} > neutral {base_minhop:.2f})"
    elif best_val >= 0.25:
        verdict = f"PARTIAL: best '{best}' -> {best_val:.2f} (neutral {base_minhop:.2f}); primitive helps, gap not closed"
    else:
        verdict = f"IMPOSSIBILITY: no arm exceeds 0.25 (best '{best}' {best_val:.2f}); aggressive-compressor loss is primitive-invariant -> offload (Pillar-4)"
    res["verdict"] = verdict
    json.dump(res, open(f"out/pilot/rescue_ceiling_{tag}.json", "w"), indent=2)
    print(f"=== RESCUE CEILING ({tag}, numeric, pin_oracle re-injection) ===")
    for s in STANCES:
        d = res["stances"][s]
        print(f"  {s:>9}  carry-surv/hop={d['carry_survival_by_hop']}  "
              f"min-hop={d['min_hop_carry_survival']:.3f} CI{d['min_hop_ci']}")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
