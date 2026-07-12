# Tier-1 #3 (design doc S8): REAL-FRAMEWORK validation. Rebut "the law is an artifact of
# our bespoke relay.py loop" by re-implementing the handoff chain in LangGraph -- an
# independent, widely used multi-agent orchestration framework. Each hop is a real
# StateGraph node that reads shared state, calls the SAME model to summarize under budget,
# and writes the handoff state for the next node. We check that (a) the fact still decays
# roughly exponentially with hop count and (b) the FRAMING effect (persona << neutral)
# still appears -- i.e. the phenomena are not harness artifacts.
#
# Node prompts are LangGraph-native ("you are a node in a pipeline ... INCOMING STATE")
# and use a SEPARATE cache, so this is an independent reproduction, not a cache replay of
# relay.py. Hops are keyed independently of total k (temp-0 greedy -> hop i depends only on
# hops < i), so longer chains reuse shorter-chain hops.
import hashlib, json, os, sys
from typing import TypedDict
from langgraph.graph import StateGraph, END
from facts import make_facts
from grade import grade
from relay import _truncate, _ROLES, _planted
from run import _build_backend
from analyze import survival_curve, fit_tau, empirical_chance, bootstrap_tau_ci


class RelayState(TypedDict):
    carry: str
    answer: str


def _node_prompt(carry: str, budget: int, mode: str, hop: int) -> str:
    keep = (f"Keep only what matters most, in at most {budget} words — you cannot keep "
            f"everything. Start your reply with 'CARRY:'.")
    if mode == "persona":
        me = _ROLES[hop % len(_ROLES)]
        nxt = _ROLES[(hop + 1) % len(_ROLES)]
        head = (f"You are {me}, a node in a LangGraph multi-agent pipeline. Write the "
                f"handoff state for the next node, {nxt}.")
    else:  # neutral handoff
        head = ("You are a node in a LangGraph pipeline. Write the handoff state that the "
                "next node will read in place of the current state.")
    return f"{head}\nINCOMING STATE: {carry}\n{keep}"


def build_app(k: int, backend, iid_base: str, query: str, budget: int, mode: str):
    g = StateGraph(RelayState)

    def make_agent(i):
        def node(state):
            p = _node_prompt(state["carry"], budget, mode, i)
            msg = backend.generate(p, item_id=f"{iid_base}:h{i}", max_tokens=budget * 3 + 16)
            return {"carry": _truncate(msg, budget)}
        return node

    prev = None
    for i in range(k):
        name = f"agent{i}"
        g.add_node(name, make_agent(i))
        if prev is None:
            g.set_entry_point(name)
        else:
            g.add_edge(prev, name)
        prev = name

    def probe(state):
        p = f"CARRY: {state['carry']}\nQUESTION: {query}\nAnswer concisely."
        ans = backend.generate(p, item_id=f"{iid_base}:q{k}", max_tokens=64)
        return {"answer": ans}

    g.add_node("probe", probe)
    if prev is None:
        g.set_entry_point("probe")
    else:
        g.add_edge(prev, "probe")
    g.add_edge("probe", END)
    return g.compile()


def run_cell(backend, facts, k, mode, budget, m_facts, seed=0):
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    rows = []
    for f in facts:
        pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
        distractors = tuple(pool[:max(0, m_facts - 1)])
        planted = _planted(f, distractors, seed)
        iid_base = "rf:" + hashlib.sha1(
            f"{f.fact_id}|{mode}|{budget}|{m_facts}|{seed}".encode()).hexdigest()[:14]
        app = build_app(k, backend, iid_base, f.query, budget, mode)
        out = app.invoke({"carry": planted, "answer": ""})
        rows.append({"model": backend.name, "condition": mode, "k": k,
                     "fact_id": f.fact_id, "ftype": f.ftype, "seed": seed,
                     "correct": grade(f, out["answer"], "actionable")})
    return rows


def _chance(backend, facts, m_facts):
    # nofact baseline: probe with no planted info -> empirical guess rate
    rows = []
    for f in facts:
        iid = "rfnf:" + hashlib.sha1(f"{f.fact_id}".encode()).hexdigest()[:12]
        p = f"CARRY: (no prior information provided)\nQUESTION: {f.query}\nAnswer concisely."
        ans = backend.generate(p, item_id=iid, max_tokens=64)
        rows.append({"condition": "nofact", "ftype": f.ftype, "correct": grade(f, ans, "actionable")})
    return sum(r["correct"] for r in rows) / len(rows)


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="40",
         ks="0,1,2,4", budget="25", m_facts="8", conds="handoff,persona"):
    n = int(n); budget = int(budget); m_facts = int(m_facts)
    ks_list = [int(x) for x in str(ks).split(",")]
    cond_list = str(conds).split(",")
    facts = make_facts(n, seed=0)
    tag = model.split('/')[-1]
    backend = _build_backend(provider, model, f"data/cache_realframe_{tag}.json")
    chance = _chance(backend, facts, m_facts)
    os.makedirs("out/pilot", exist_ok=True)
    rowf = open(f"out/pilot/realframe_rows_{tag}.jsonl", "w")
    res = {"model": model, "framework": "langgraph", "n": n, "budget": budget,
           "m_facts": m_facts, "chance": chance, "conditions": {}}
    all_rows = []
    for c in cond_list:
        crows = []
        for k in ks_list:
            crows += run_cell(backend, facts, k, c, budget, m_facts)
        for r in crows:
            rowf.write(json.dumps(r) + "\n")
        all_rows += crows
        kk, S = survival_curve(crows, c)
        fit = fit_tau(kk, S, chance)
        ci = bootstrap_tau_ci(crows, c, chance)
        res["conditions"][c] = {"tau": fit["tau"], "tau_ci": [ci["lo"], ci["hi"]],
                                "r2": fit["r2"], "f": fit["f"], "S": S}
        json.dump(res, open(f"out/pilot/realframe_{tag}.json", "w"), indent=2)
        print(f"[langgraph] {c:9s} tau={fit['tau']:.2f} CI=[{ci['lo']:.2f},{ci['hi']:.2f}] "
              f"r2={fit['r2']:.2f} S={[round(x,2) for x in S]}", flush=True)
    rowf.close()
    print(f"chance={chance:.3f}")
    if "handoff" in res["conditions"] and "persona" in res["conditions"]:
        th = res["conditions"]["handoff"]["tau"]; tp = res["conditions"]["persona"]["tau"]
        print(f"FRAMING in real framework: persona {tp:.2f} vs handoff {th:.2f} "
              f"-> {'REPRODUCED (persona faster)' if tp < th else 'NOT reproduced'}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
