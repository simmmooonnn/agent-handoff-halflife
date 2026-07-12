# Confound check for the itemize result (cached hop outputs, no GPU): is itemize's
# near-zero decay (S(4)=0.87 vs faithful 0.58) bought by EXCEEDING the word budget
# (longer outputs = a budget increase in disguise), or is it a genuine FORMAT effect
# at comparable length?
import hashlib, json, sys
from statistics import mean
from facts import make_facts
from relay import item_id_for, TEMPLATE_VERSION
from run import FILLER

CONDS = ["handoff", "faithful", "neutral_long", "itemize", "manifest", "faithman"]

def main(cache_path="data/cache_pilot_Qwen2.5-7B-Instruct.json", n="100",
         budget="25", m_facts="8"):
    n = int(n); budget = int(budget); m_facts = int(m_facts)
    cache = json.load(open(cache_path, encoding="utf-8"))
    prefix = "hf/" + cache_path.split("cache_pilot_")[1].split(".json")[0]
    facts = make_facts(n, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    print(f"{'cond':>13} {'hops':>6} {'mean_w':>7} {'p50':>5} {'p90':>5} {'%>budget':>9} {'%>2xbudget':>11}")
    for cond in CONDS:
        lens = []
        for f in facts:
            pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
            distractors = tuple(pool[:m_facts - 1])
            dctx = ",".join(d.fact_id for d in distractors)
            ctx = hashlib.sha1(f"{FILLER}|{TEMPLATE_VERSION}|{dctx}".encode()).hexdigest()[:8]
            for k in (1, 2, 4):
                iid = item_id_for(f, k, cond, budget, "light", 0)
                for i in range(k):
                    key = f"{prefix}:{iid}:{ctx}:h{i}"
                    if key in cache:
                        lens.append(len(cache[key].split()))
        if lens:
            lens.sort()
            p50 = lens[len(lens) // 2]
            p90 = lens[int(0.9 * len(lens))]
            over = 100.0 * sum(1 for x in lens if x > budget) / len(lens)
            over2 = 100.0 * sum(1 for x in lens if x > 2 * budget) / len(lens)
            print(f"{cond:>13} {len(lens):>6} {mean(lens):>7.1f} {p50:>5} {p90:>5} {over:>8.1f}% {over2:>10.1f}%")
        else:
            print(f"{cond:>13} {'MISS':>6}")

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
