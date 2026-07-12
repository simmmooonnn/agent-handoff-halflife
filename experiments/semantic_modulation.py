# SEMANTIC MODULATION test (prereg: out/pilot/semantic_modulation_predictions.json,
# commit 77c7a67). On gpt-5.5, re-inject a fact at the front of the carry EVERY hop
# (pin_oracle style) under NEUTRAL compression, varying only the INJECTION FORM:
#   bare          declarative           (baseline)
#   constraint    number->prohibition   (MODULATION: move fragile type onto robust type)
#   salience      'CRITICAL preserve..'  (importance-label control; isolates role vs label)
#   bare_faithful bare + faithful stance (E5 lever cross-reference)
# Measures answer-token carry-survival (== answered) per type at probe hops -> directly
# tests whether the DIGIT rides up, not just the constraint skeleton. Own cache, no API
# reuse. TYPES = fragile/mid types only (negation is already constraint-form).
import hashlib, json, os, sys
import numpy as np
from facts import make_facts
from grade import grade, _wb
from relay import _planted, _agent_prompt, _compress_prompt, _truncate, _LOAD_REPS
from run import FILLER, _build_backend

M, B_ri, K, budget = 24, 4, 16, 25
PROBE_KS = [4, 8, 16]
TYPES = ["numeric", "entity", "preference"]
ARMS = ["bare", "constraint", "salience", "bare_faithful"]


def transcode(fact, form):
    """Return the re-injected TEXT for this fact under the given injection form."""
    s = fact.statement
    if form in ("bare", "bare_faithful"):
        return s
    if form == "salience":
        return f"CRITICAL, preserve verbatim: {s}"
    # form == "constraint": re-encode into a prohibition binding the answer token
    t = fact.ftype
    if t == "numeric":
        # "The <subj> is <val>."
        subj = s[len("The "):].rsplit(" is ", 1)[0]
        val = fact.answer
        return (f"Reject as an error any {subj} other than {val}; "
                f"the {subj} must be exactly {val}.")
    if t == "entity":
        subj = s[len("The "):].rsplit(" is ", 1)[0]
        name = fact.answer
        return (f"Reject anyone other than {name} as the {subj}; "
                f"the {subj} must be {name}.")
    if t == "preference":
        # "For <aspect>, the user prefers <choice>."
        aspect = s[len("For "):].split(", the user prefers", 1)[0]
        choice = fact.answer
        return (f"Reject any option other than {choice} for {aspect}; "
                f"the required {aspect} is {choice}.")
    return s


def compress_prompt(carry, work, arm, hop):
    if arm == "bare_faithful":
        return _compress_prompt(carry, work, budget, "faithful", hop, K)
    return _agent_prompt(carry, work, None, budget=budget)


def main(provider="openai", model="gpt-5.5"):
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_mod_{tag}.json")
    facts = make_facts(100, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    types_cycle = sorted(by_type)
    work = " " + FILLER * _LOAD_REPS["light"]
    os.makedirs("out/pilot", exist_ok=True)

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

    def gen(prompt, label, mt):
        h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        return backend.generate(prompt, item_id=f"mod:{label}:{h}", max_tokens=mt)

    # mat[arm][type][fid][hop] = correct ; surv same for carry-presence
    mat = {a: {t: {} for t in TYPES} for a in ARMS}
    surv = {a: {t: {} for t in TYPES} for a in ARMS}
    probe_facts = [f for f in facts if f.ftype in TYPES]
    for fi, probe in enumerate(probe_facts):
        distractors = mixed_distractors(probe)
        chain = [probe] + list(distractors)
        for arm in ARMS:
            inj = transcode(probe, arm)
            # re-inject the OTHER pinned facts as bare (only the probe is modulated); the
            # probe is index 0 and always pinned, matching pin_oracle.
            pinned_texts = [inj] + [chain[i].statement for i in range(1, B_ri)]
            carry = _planted(probe, distractors, seed=0)
            row, srow = {}, {}
            for h in range(1, K + 1):
                carry = " ".join(pinned_texts) + " " + carry
                p = compress_prompt(carry, work, arm, h - 1)
                msg = gen(p, f"{arm}:{probe.fact_id}:h{h}", budget * 3 + 16)
                carry = _truncate(msg, budget)
                if h in PROBE_KS:
                    srow[h] = 1 if _wb(probe.answer) in _wb(carry) else 0
                    q = _agent_prompt(carry, "", probe.query)
                    resp = gen(q, f"{arm}:{probe.fact_id}:q{h}", 64)
                    row[h] = grade(probe, resp, "actionable")
            mat[arm][probe.ftype][probe.fact_id] = row
            surv[arm][probe.ftype][probe.fact_id] = srow
        with open(f"out/pilot/mod_progress_{tag}.txt", "w") as fh:
            fh.write(f"done {fi + 1}/{len(probe_facts)}\n")

    def worst_hop_by_type(m, t, idx=None):
        fids = sorted(m[t])
        sel = fids if idx is None else [fids[i] for i in idx]
        return min(np.mean([m[t][fid][k] for fid in sel]) for k in PROBE_KS)

    def estimand(m, idx=None, kind="strictmin"):
        vals = [worst_hop_by_type(m, t, idx.get(t) if idx else None) for t in TYPES]
        return float(min(vals)) if kind == "strictmin" else float(np.mean(vals))

    rng = np.random.default_rng(0)
    sizes = {t: len(mat["bare"][t]) for t in TYPES}
    res = {"model": model, "arms": {}, "numeric_worst_hop": {}, "answered": {}}
    for a in ARMS:
        res["arms"][a] = {t: round(worst_hop_by_type(surv[a], t), 3) for t in TYPES}
        res["numeric_worst_hop"][a] = round(worst_hop_by_type(surv[a], "numeric"), 3)
        res["answered"][a] = {t: round(worst_hop_by_type(mat[a], t), 3) for t in TYPES}

    def paired_ci(a, b, t="numeric", kind="worsthop"):
        d = []
        for _ in range(2000):
            idx = {tt: rng.integers(0, sizes[tt], size=sizes[tt]) for tt in TYPES}
            if kind == "worsthop":
                d.append(worst_hop_by_type(surv[a], t, idx[t]) - worst_hop_by_type(surv[b], t, idx[t]))
            else:
                d.append(estimand(surv[a], idx, "strictmin") - estimand(surv[b], idx, "strictmin"))
        return round(float(np.mean(d)), 3), [round(float(np.percentile(d, 2.5)), 3),
                                             round(float(np.percentile(d, 97.5)), 3)]
    cons = {}
    for (a, b) in [("constraint", "bare"), ("constraint", "salience"),
                   ("constraint", "bare_faithful"), ("bare_faithful", "bare")]:
        pt, ci = paired_ci(a, b, "numeric")
        cons[f"{a}_minus_{b}_numeric"] = {"point": pt, "ci": ci, "sig": "CI-excl-0" if ci[0] > 0 else "NS"}
    res["contrasts_numeric_carry_survival"] = cons
    cnum = res["numeric_worst_hop"]["constraint"]
    solve = (cnum >= 0.50 and cons["constraint_minus_bare_numeric"]["ci"][0] > 0
             and cons["constraint_minus_salience_numeric"]["ci"][0] > 0)
    res["verdict"] = ("SOLVE: semantic modulation lifts worst-case numeric survival >=0.5, CI-excl bare AND salience"
                      if solve else ("PARTIAL" if cnum >= 0.25 else "CAVEAT_REALIZED: digit drops even inside a constraint"))
    json.dump(res, open(f"out/pilot/semantic_modulation_{tag}.json", "w"), indent=2)
    print(f"=== SEMANTIC MODULATION ({tag}) numeric worst-hop carry-survival ===")
    for a in ARMS:
        print(f"  {a:>14}: numeric={res['numeric_worst_hop'][a]:.3f}  by_type={res['arms'][a]}")
    print("  contrasts (numeric carry-survival):")
    for k, v in cons.items():
        print(f"    {k:>34}: {v['point']:+.3f} {v['ci']} {v['sig']}")
    print("VERDICT:", res["verdict"])


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
