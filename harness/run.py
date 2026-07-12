import argparse, json, os
from facts import make_facts
from grade import grade
from relay import run_chain
from backends import MockBackend, CachedBackend

FILLER = "The quarterly report covers logistics, vendor SLAs, and regional rollout notes."

def run_sweep(backend, facts, ks, conditions, *, budget, load, filler, seeds, level, m_facts=1):
    # m_facts = total facts carried per chain (1 probe + m_facts-1 same-type distractors).
    # Information pressure: a tight per-hop budget cannot retain all m_facts, forcing loss.
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    rows = []
    for f in facts:
        pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
        distractors = tuple(pool[:max(0, m_facts - 1)])
        for cond in conditions:
            for k in ks:
                for s in seeds:
                    resp = run_chain(backend, f, k, cond, budget=budget,
                                     load=load, filler=filler, seed=s, distractors=distractors)
                    rows.append({"model": backend.name, "condition": cond, "k": k,
                                 "fact_id": f.fact_id, "ftype": f.ftype, "seed": s,
                                 "correct": grade(f, resp, level)})
    return rows

def _build_backend(provider, model, cache):
    if provider == "mock":
        inner = MockBackend("mock", lambda p: next((l for l in p.splitlines()
                            if l.startswith("CARRY:")), "CARRY:"))
    elif provider in ("hf", "openai", "anthropic"):
        from model_backends import build_model_backend   # Task 7
        inner = build_model_backend(provider, model)
    else:
        raise ValueError(provider)
    return CachedBackend(inner, cache)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--ks", default="0,1,2,4,8,16")
    ap.add_argument("--conditions", default="handoff,longctx,verbatim")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--load", default="light")
    ap.add_argument("--m-facts", dest="m_facts", type=int, default=8)
    ap.add_argument("--level", default="actionable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="data/signals_cache.json")
    a = ap.parse_args()
    facts = make_facts(a.n, seed=0)
    backend = _build_backend(a.provider, a.model, a.cache)
    rows = run_sweep(backend, facts, [int(x) for x in a.ks.split(",")],
                     a.conditions.split(","), budget=a.budget, load=a.load,
                     filler=FILLER, seeds=[int(x) for x in a.seeds.split(",")],
                     level=a.level, m_facts=a.m_facts)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {a.out}")

if __name__ == "__main__":
    main()
