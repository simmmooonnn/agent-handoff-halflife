# Phase 2 LIVE validation of the re-injection method on a real model. We run the handoff
# chain at K hops under several re-injection schedules and measure the probe's ACTUAL final
# survival per fact type. This validates (a) the mechanism -- restating a fact verbatim
# resets its decay, so survival rises with re-injection frequency on the real model -- and
# (b) the method -- because fact types decay at different measured rates, a tau-aware
# allocation of a fixed re-injection budget beats a uniform one (reconstructed from these
# live per-(ftype,cost) survival measurements).
import json, os, sys
from facts import make_facts
from grade import grade
from relay import run_chain
from run import FILLER, _build_backend


def _survival(backend, facts, K, schedule, budget, m_facts, seed=0):
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    rows = []
    for f in facts:
        pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
        distractors = tuple(pool[:max(0, m_facts - 1)])
        resp = run_chain(backend, f, K, "handoff", budget=budget, load="light",
                         filler=FILLER, seed=seed, distractors=distractors,
                         reinject_hops=tuple(schedule))
        rows.append({"ftype": f.ftype, "correct": grade(f, resp, "actionable")})
    return rows


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="60", K="8",
         budget="25", m_facts="8"):
    n = int(n); K = int(K); budget = int(budget); m_facts = int(m_facts)
    facts = make_facts(n, seed=0)
    tag = model.split('/')[-1]
    backend = _build_backend(provider, model, f"data/cache_reinject_{tag}.json")
    # schedules over K=8: never, every-4, every-2, every-1 (cost = len)
    schedules = {"never": [], "every4": list(range(4, K + 1, 4)),
                 "every2": list(range(2, K + 1, 2)), "every1": list(range(1, K + 1))}
    ftypes = sorted({f.ftype for f in facts})
    out = {"model": model, "K": K, "budget": budget, "m_facts": m_facts, "schedules": {}}
    os.makedirs("out/pilot", exist_ok=True)
    print(f"=== live re-injection eval (handoff, K={K}, budget={budget}, M={m_facts}) ===")
    hdr = "  schedule   cost " + "".join(f"{ft:>11s}" for ft in ftypes) + "   min  mean"
    print(hdr)
    for name, sch in schedules.items():
        rows = _survival(backend, facts, K, sch, budget, m_facts)
        per = {}
        for ft in ftypes:
            v = [r["correct"] for r in rows if r["ftype"] == ft]
            per[ft] = sum(v) / len(v) if v else 0.0
        cost = len(sch)  # per-fact re-injections
        mn = min(per.values()); mean = sum(per.values()) / len(per)
        out["schedules"][name] = {"cost_per_fact": cost, "hops": sch, "per_ftype": per,
                                  "min": mn, "mean": mean}
        json.dump(out, open(f"out/pilot/reinject_{tag}.json", "w"), indent=2)
        print(f"  {name:9s} {cost:4d} " + "".join(f"{per[ft]:11.2f}" for ft in ftypes)
              + f"   {mn:.2f}  {mean:.2f}", flush=True)
    # mechanism check: does survival rise monotonically with re-injection frequency?
    seq = [out["schedules"][s]["mean"] for s in ["never", "every4", "every2", "every1"]]
    mono = all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))
    print(f"\nmechanism: mean survival vs freq = {[round(x,2) for x in seq]} "
          f"-> {'MONOTONIC increase (re-injection works)' if mono else 'non-monotonic'}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
