# HANDOFFBENCH (release artifact, paper contribution C): a reusable diagnostic that turns
# this paper's measurements into a tool. It (1) aggregates every measured handoff half-life
# tau across the study (size ladder, budget sweep, agent-boundary framings, fact types) into
# a single manifest, and (2) exposes the planning primitives a practitioner needs:
#   max_safe_depth(tau,f,S0,theta) -- how many handoffs a fact survives before dropping to theta
#   reinjection_period(...)        -- how often to restate it to hold survival >= theta
#   plan_reinjection(facts,K,theta)-- a full tau-aware schedule for a mixed-fact pipeline
# Run `python handoffbench.py build` to (re)generate out/handoffbench_manifest.json from the
# result JSONs, or `python handoffbench.py` to print the manifest + a worked example.
import json, glob, math, os, sys

TAG = "Qwen2.5-7B-Instruct"


def max_safe_depth(tau, f, S0, theta):
    """Handoffs a fact survives before its usability drops to theta (exponential survival)."""
    if theta <= f:
        return math.inf            # floor already >= theta: never lost
    if theta >= S0:
        return 0.0
    return tau * math.log((S0 - f) / (theta - f))


def reinjection_period(tau, f, S0, theta):
    """Integer handoff period at which to restate a fact to keep survival >= theta."""
    a = max_safe_depth(tau, f, S0, theta)
    return math.inf if a == math.inf else max(int(math.floor(a)), 1)


def plan_reinjection(facts, K, theta):
    """facts: list of dicts with tau,f,S0,label. Returns a per-fact re-injection schedule and
    the total cost -- the tau-aware policy from scheduler.py, ready to drive a real pipeline."""
    plan = {}
    total = 0
    for fc in facts:
        p = reinjection_period(fc["tau"], fc["f"], fc["S0"], theta)
        hops = [] if p == math.inf else list(range(p, K + 1, p))
        plan[fc.get("label", fc["tau"])] = {"period": (None if p == math.inf else p), "hops": hops}
        total += len(hops)
    return {"schedule": plan, "total_reinjections": total, "K": K, "theta": theta}


def build_manifest():
    m = {"model": TAG, "decay_model": "S(k)=f+(S0-f)exp(-k/tau)", "measurements": {}}
    # size ladder
    ladder = {}
    for fp in sorted(glob.glob("out/pilot/verdict_Qwen2.5-*.json")):
        d = json.load(open(fp))
        tag = os.path.basename(fp).split("Qwen2.5-")[1].split("-Instruct")[0]
        if "handoff" in d:
            ladder[tag] = round(d["handoff"]["tau"], 3)
    m["measurements"]["size_ladder_tau"] = ladder
    # budget sweep
    bp = f"out/pilot/budget_sweep_{TAG}.json"
    if os.path.exists(bp):
        d = json.load(open(bp))
        m["measurements"]["budget_tau"] = {b["budget"]: round(b["tau"], 3) for b in d["budgets"]}
    # framings
    dp = f"out/pilot/disambig_{TAG}.json"
    if os.path.exists(dp):
        C = json.load(open(dp))["conditions"]
        m["measurements"]["framing_tau"] = {
            k: {"tau": round(v["tau"], 3), "f": round(v["f"], 3), "S0": round(v["S"][0], 3)}
            for k, v in C.items()}
    # fact types
    pp = f"out/pilot/perftype_{TAG}_handoff.json"
    if os.path.exists(pp):
        d = json.load(open(pp))
        m["measurements"]["ftype_tau"] = {
            ft: {"tau": round(v["tau"], 3), "f": round(v["f"], 3), "S0": round(v["S"][0], 3),
                 "chance": round(v["chance"], 3)}
            for ft, v in d["by_ftype"].items()}
    os.makedirs("out", exist_ok=True)
    json.dump(m, open("out/handoffbench_manifest.json", "w"), indent=2)
    return m


def main(cmd=""):
    if cmd == "build":
        m = build_manifest()
        print("built out/handoffbench_manifest.json")
    else:
        p = "out/handoffbench_manifest.json"
        m = json.load(open(p)) if os.path.exists(p) else build_manifest()
    print(f"=== HANDOFFBENCH manifest ({m['model']}) ===")
    print(f"decay model: {m['decay_model']}")
    for k, v in m["measurements"].items():
        print(f"\n[{k}]")
        print(" ", json.dumps(v))
    # worked example: mixed-fact pipeline, tau-aware plan
    ft = m["measurements"].get("ftype_tau", {})
    facts = [{"label": name, "tau": v["tau"], "f": v["f"], "S0": v["S0"]}
             for name, v in ft.items() if (v["S0"] - v["chance"]) >= 0.2]
    if facts:
        print("\n=== worked example: tau-aware re-injection plan (K=16, theta=0.5) ===")
        plan = plan_reinjection(facts, K=16, theta=0.5)
        for lab, s in plan["schedule"].items():
            print(f"  {lab:10s} restate every {s['period']} hops  (hops {s['hops']})")
        print(f"  total re-injections: {plan['total_reinjections']}")
        for fc in facts:
            print(f"  max_safe_depth({fc['label']}) = "
                  f"{max_safe_depth(fc['tau'], fc['f'], fc['S0'], 0.5):.2f} handoffs before S<0.5")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
