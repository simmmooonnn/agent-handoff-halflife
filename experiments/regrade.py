# R1.2 grader sensitivity: re-score every cached probe answer under the
# word-boundary grader (grade.py boundary=True, the new default) vs the legacy
# strip-everything containment (boundary=False), WITHOUT re-running any model.
#
# This works because CachedBackend keys on (backend.name, item_id) only -- the
# probe answer for a row is at "{name}:{iid}:{ctx}:q", and iid/ctx are pure
# functions of the run config. Self-check: the legacy re-grade must reproduce
# the stored rows_*.jsonl verdicts bit-for-bit wherever those files exist; any
# mismatch means the key reconstruction is wrong and the sweep is reported as
# UNVERIFIED rather than silently trusted.
#
# Run on Anvil (caches live there); CPU-only, no GPU needed.
import hashlib, json, os, sys
from statistics import mean
from facts import make_facts
from grade import grade
from relay import item_id_for, TEMPLATE_VERSION
from run import FILLER

Q7 = "Qwen/Qwen2.5-7B-Instruct"
LADDER = [f"Qwen/Qwen2.5-{s}-Instruct" for s in
          ("0.5B", "1.5B", "3B", "7B", "14B", "32B")] + \
         ["mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-3-mini-4k-instruct",
          "microsoft/Phi-3-medium-4k-instruct"]
XFAM = ["mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-3-medium-4k-instruct"]

# (sweep, model, conds, ks, n, budgets, m_facts, cache_tpl, rows_file_or_None)
def _tag(m):
    return m.split("/")[-1]

SWEEPS = (
    [("pilot", m, ["handoff", "longctx", "verbatim"], [0, 1, 2, 4, 8, 16], 100, [25],
      f"data/cache_pilot_{_tag(m)}.json", f"out/pilot/rows_{_tag(m)}.jsonl")
     for m in LADDER] +
    [("framing", Q7,
      ["persona", "selfsumm", "handoff", "neutral_long", "node", "rolecont",
       "distrust", "faithful", "nofact"], [0, 1, 2, 4], 100, [25],
      f"data/cache_pilot_{_tag(Q7)}.json", f"out/pilot/framing_rows_{_tag(Q7)}.jsonl")] +
    [("framing", m, ["node", "rolecont", "handoff", "faithful", "manifest", "nofact"],
      [0, 1, 2, 4], 100, [25],
      f"data/cache_pilot_{_tag(m)}.json", f"out/pilot/framing_rows_{_tag(m)}.jsonl")
     for m in XFAM] +
    [("protocol", Q7,
      ["faithful", "faithman", "handoff", "itemize", "manifest", "neutral_long",
       "nofact"], [0, 1, 2, 4], 100, [25],
      f"data/cache_pilot_{_tag(Q7)}.json", f"out/pilot/protocol_rows_{_tag(Q7)}.jsonl")] +
    [("newstance", Q7, ["ledger", "editor", "link"], [0, 1, 2, 4], 100, [25],
      f"data/cache_pilot_{_tag(Q7)}.json", None),
     ("newstance12", Q7, ["ledger"], [0, 1, 2, 4], 100, [12],
      f"data/cache_pilot_{_tag(Q7)}.json", None),
     ("deepprobe", Q7, ["ledger", "editor", "link", "manifest", "faithman",
                        "neutral_long"], [8], 100, [25],
      f"data/cache_pilot_{_tag(Q7)}.json", None),
     ("deepprobe12", Q7, ["ledger"], [8], 100, [12],
      f"data/cache_pilot_{_tag(Q7)}.json", None),
     ("deepprobe16", Q7, ["ledger", "manifest"], [16], 100, [25],
      f"data/cache_pilot_{_tag(Q7)}.json", None),
     ("budget", Q7, ["handoff"], [0, 1, 2, 4], 100, [8, 16, 50],
      f"data/cache_pilot_{_tag(Q7)}.json", None)] +
    [("budget", m, ["handoff"], [0, 1, 2, 4], 40, [8, 16, 25, 50],
      f"data/cache_pilot_{_tag(m)}.json", None) for m in XFAM]
)

LOAD, SEED, M_FACTS = "light", 0, 8


def probe_key(name, fact, k, cond, budget, distractors):
    iid = item_id_for(fact, k, cond, budget, LOAD, SEED)
    dctx = ",".join(d.fact_id for d in distractors)
    ctx = hashlib.sha1(f"{FILLER}|{TEMPLATE_VERSION}|{dctx}".encode()).hexdigest()[:8]
    return f"{name}:{iid}:{ctx}:q"


def main():
    facts = make_facts(100, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)

    def dis(f):
        pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
        return tuple(pool[:M_FACTS - 1])

    stored = {}  # rows_file -> {(cond,k,fact_id): correct}
    caches = {}
    out_rows, cells, total, flips, miss = [], {}, 0, 0, 0
    selfcheck = {"ok": 0, "bad": 0, "files": {}}

    for sweep, model, conds, ks, n, budgets, cache_f, rows_f in SWEEPS:
        if cache_f not in caches:
            caches[cache_f] = (json.load(open(cache_f))
                               if os.path.exists(cache_f) else None)
        cache = caches[cache_f]
        if cache is None:
            print(f"[skip] {sweep}/{_tag(model)}: no cache {cache_f}")
            continue
        if rows_f and rows_f not in stored and os.path.exists(rows_f):
            stored[rows_f] = {(r["condition"], r["k"], r["fact_id"]): r["correct"]
                              for r in map(json.loads, open(rows_f))}
        name = f"hf/{_tag(model)}"
        for budget in budgets:
            for cond in conds:
                for k in ks:
                    olds, news = [], []
                    for f in facts[:n]:
                        key = probe_key(name, f, k, cond, budget, dis(f))
                        resp = cache.get(key)
                        if resp is None:
                            miss += 1
                            continue
                        o = grade(f, resp, "actionable", boundary=False)
                        w = grade(f, resp, "actionable", boundary=True)
                        olds.append(o); news.append(w)
                        total += 1
                        flips += (o != w)
                        st = stored.get(rows_f, {}).get((cond, k, f.fact_id))
                        if st is not None:
                            good = (st == o)
                            selfcheck["ok" if good else "bad"] += 1
                            fc = selfcheck["files"].setdefault(rows_f, [0, 0])
                            fc[0 if good else 1] += 1
                        out_rows.append({"sweep": sweep, "model": name,
                                         "condition": cond, "k": k, "budget": budget,
                                         "fact_id": f.fact_id, "ftype": f.ftype,
                                         "correct_old": o, "correct_new": w})
                    if olds:
                        cells[f"{sweep}|{_tag(model)}|{cond}@{budget}|k{k}"] = {
                            "n": len(olds), "S_old": round(mean(olds), 4),
                            "S_new": round(mean(news), 4),
                            "flips": sum(o != w for o, w in zip(olds, news))}

    os.makedirs("out/pilot", exist_ok=True)
    with open("out/pilot/regrade_rows.jsonl", "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    worst = sorted(cells.items(), key=lambda kv: -abs(kv[1]["S_new"] - kv[1]["S_old"]))
    summary = {"total_graded": total, "cache_misses": miss,
               "flip_count": flips, "flip_rate": round(flips / max(total, 1), 5),
               "selfcheck_ok": selfcheck["ok"], "selfcheck_bad": selfcheck["bad"],
               "selfcheck_files": selfcheck["files"],
               "max_cell_shift": (round(abs(worst[0][1]["S_new"] - worst[0][1]["S_old"]), 4)
                                  if worst else None),
               "cells": cells}
    json.dump(summary, open("out/pilot/regrade_summary.json", "w"), indent=1)
    print(f"graded {total} cached answers, {miss} misses")
    print(f"self-check vs stored rows: {selfcheck['ok']} match, "
          f"{selfcheck['bad']} MISMATCH {'<-- KEY RECONSTRUCTION BROKEN' if selfcheck['bad'] else '(reconstruction verified)'}")
    print(f"verdict flips old->new: {flips} ({100 * flips / max(total, 1):.3f}%)")
    print("top cell shifts |S_new - S_old|:")
    for cname, c in worst[:10]:
        print(f"  {cname:55} S {c['S_old']:.3f} -> {c['S_new']:.3f}  ({c['flips']} flips / {c['n']})")


if __name__ == "__main__":
    main()
