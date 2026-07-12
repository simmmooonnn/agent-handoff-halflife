# E1 post-hoc mechanism check (no GPU; reads the response cache).
# The coupling gate FAILED: re-injecting up to 4 facts/hop did not hurt the bystander,
# violating the effective-budget arithmetic (B_eff = B - m*w) by ~8x. Hypothesis: the
# soft word budget is ELASTIC -- under injected pressure the model exceeds the "at most
# N words" instruction instead of evicting content. Test: reconstruct the cache keys of
# every hop message in the E1 grid and compare output word-lengths across injection load.
import hashlib, json, sys
from statistics import mean
from facts import make_facts
from relay import item_id_for, TEMPLATE_VERSION
from run import FILLER

def main(cache_path="data/cache_pilot_Qwen2.5-7B-Instruct.json", n="100", m_facts="8"):
    n = int(n); m_facts = int(m_facts)
    cache = json.load(open(cache_path, encoding="utf-8"))
    prefix = "hf/" + cache_path.split("cache_pilot_")[1].split(".json")[0]
    facts = make_facts(n, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    print(f"{'budget':>6} {'m':>2} {'hops_found':>10} {'mean_words':>10} {'p90':>6} {'%>budget':>9}")
    out = {}
    for budget in (25, 16):
        for m in (0, 1, 2, 4):
            lens = []
            for f in facts:
                pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
                distractors = tuple(pool[:m_facts - 1])
                inj = [d.statement for d in distractors[:m]] if m > 0 else None
                for k in (1, 2, 4):
                    iid = item_id_for(f, k, "handoff", budget, "light", 0)
                    reinject = set(range(1, k + 1)) if m > 0 else set()
                    dctx = ",".join(d.fact_id for d in distractors)
                    ri_sig = ("|ri" + ",".join(map(str, sorted(reinject)))) if reinject else ""
                    if reinject and inj is not None:
                        ri_sig += "|rf" + hashlib.sha1(" ".join(inj).encode()).hexdigest()[:8]
                    ctx = hashlib.sha1(f"{FILLER}|{TEMPLATE_VERSION}|{dctx}{ri_sig}".encode()).hexdigest()[:8]
                    for i in range(k):
                        key = f"{prefix}:{iid}:{ctx}:h{i}"
                        if key in cache:
                            lens.append(len(cache[key].split()))
            if lens:
                lens.sort()
                p90 = lens[int(0.9 * len(lens))]
                over = 100.0 * sum(1 for x in lens if x > budget) / len(lens)
                out[(budget, m)] = {"n": len(lens), "mean": mean(lens), "p90": p90, "pct_over": over}
                print(f"{budget:>6} {m:>2} {len(lens):>10} {mean(lens):>10.1f} {p90:>6} {over:>8.1f}%")
            else:
                print(f"{budget:>6} {m:>2} {'0 (MISS)':>10}")
    # verdict: does mean output length grow with injection load?
    for budget in (25, 16):
        if (budget, 0) in out and (budget, 4) in out:
            g = out[(budget, 4)]["mean"] - out[(budget, 0)]["mean"]
            print(f"budget={budget}: mean words inj4 - inj0 = {g:+.1f} "
                  f"-> {'ELASTIC (model exceeds soft budget under pressure)' if g > 3 else 'not elastic; nulls need another explanation'}")

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
