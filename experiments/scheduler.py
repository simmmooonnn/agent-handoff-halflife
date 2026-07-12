# METHOD (paper contribution, not just measurement): a tau-aware re-injection scheduler.
# The measurements give each fact class a decay curve S(a)=f+(S0-f)exp(-a/tau) where a is
# the number of handoffs since the fact was last seen verbatim. To keep a fact usable across
# a depth-K pipeline we RE-INJECT it (restate it verbatim -> age resets to 0), at a token
# cost. Facts decay at very different rates (measured tau spans ~3x across fact types, ~19x
# across framings), so a UNIFORM re-injection period wastes budget on slow-decaying facts and
# starves fast-decaying ones. A tau-aware scheduler re-injects each fact just before it would
# drop below a survival threshold theta, which is provably the minimum number of re-injections
# that keeps S>=theta for that fact (exponential-survival, known tau):
#     a* = tau * ln((S0-f)/(theta-f)),   n_reinject = floor(K / floor(a*)).
# We compare never / always / fixed-period(uniform) / tau-aware on a cost-vs-retention Pareto.
import json, math, os
import numpy as np


def load_classes(perftype="out/pilot/perftype_Qwen2.5-7B-Instruct_handoff.json",
                 disambig="out/pilot/disambig_Qwen2.5-7B-Instruct.json"):
    # A realistic pipeline carries facts of heterogeneous "stickiness": tau varies with the
    # fact TYPE and with the FRAMING/budget the fact is carried under. We assemble the fact
    # catalog from the EMPIRICALLY MEASURED tau spectrum (per-type handoff fits + the extreme
    # framings), spanning tau ~0.26..5.0 (x19). This heterogeneity is exactly what a uniform
    # re-injection period cannot exploit.
    classes = {}
    d = json.load(open(perftype))
    for ft, v in d["by_ftype"].items():
        S0 = v["S"][0]
        if (S0 - v["chance"]) < 0.2:      # near-chance class: info already gone, not keepable
            continue
        classes[f"type:{ft}"] = {"tau": float(v["tau"]), "f": float(v["f"]), "S0": float(S0)}
    if os.path.exists(disambig):
        dd = json.load(open(disambig))["conditions"]
        for fr in ("persona", "neutral_long"):   # the extreme framings -> tau tails
            if fr in dd:
                v = dd[fr]
                classes[f"framing:{fr}"] = {"tau": float(v["tau"]), "f": float(v["f"]),
                                            "S0": float(v["S"][0])}
    return classes


def survive(age, c):
    return c["f"] + (c["S0"] - c["f"]) * math.exp(-age / c["tau"])


def simulate(schedule, K, c):
    # schedule: set of hops (1..K) at which the fact is re-injected (age reset to 0).
    # Returns final survival S(K) and the min survival across all hops.
    age = 0
    mins = c["S0"]
    for k in range(1, K + 1):
        age += 1
        if k in schedule:
            age = 0
        s = survive(age, c)
        mins = min(mins, s)
    return survive(age, c), mins


def a_star(c, theta):
    # max age before survival drops to theta (exponential, known tau).
    if theta <= c["f"]:
        return float("inf")          # floor already >= theta: never needs re-injection
    if theta >= c["S0"]:
        return 0.0
    return c["tau"] * math.log((c["S0"] - c["f"]) / (theta - c["f"]))


def sched_fixed(period, K):
    if period <= 0 or period == float("inf"):
        return set()
    return set(range(period, K + 1, period))


def sched_tau_aware(c, K, theta):
    a = a_star(c, theta)
    if a == float("inf"):        # floor already >= theta: never needs re-injection
        return set()
    p = max(int(math.floor(a)), 1)
    return sched_fixed(p, K)


def eval_policy(classes, K, schedules):
    # Retention = each fact's WORST survival over the whole pipeline (min over hops), since a
    # fact may be queried at any point; a re-injection landing on the query hop must not
    # flatter the score. Returns (total_cost, mean-of-per-fact-worst, min-of-per-fact-worst).
    cost = sum(len(schedules[ft]) for ft in classes)
    worst = []
    for ft, c in classes.items():
        _, mn = simulate(schedules[ft], K, c)
        worst.append(mn)
    worst = np.array(worst)
    return cost, float(worst.mean()), float(worst.min()), worst


def main():
    classes = load_classes()
    K = 16
    print(f"=== tau-aware re-injection scheduler (K={K} hops) ===")
    print("fact classes (measured):")
    for ft, c in classes.items():
        print(f"  {ft:10s} tau={c['tau']:.2f} f={c['f']:.2f} S0={c['S0']:.2f}")
    tvals = [c["tau"] for c in classes.values()]
    print(f"tau heterogeneity: {min(tvals):.2f}..{max(tvals):.2f} (x{max(tvals)/min(tvals):.1f})  "
          f"<- this is what makes tau-aware beat uniform\n")

    # --- theorem sanity: per-fact minimal re-injections to hold S>=theta ---
    theta = 0.5
    print(f"minimum re-injections to keep S>=theta={theta} over K={K} (theorem):")
    for ft, c in classes.items():
        a = a_star(c, theta)
        p = max(int(math.floor(a)), 1)
        n = len(sched_fixed(p, K))
        print(f"  {ft:10s} a*={a:.2f} -> re-inject every {p} hops -> {n} injections")

    # --- Pareto: cost vs retention ---
    # tau-aware: sweep theta -> per-class schedules guaranteeing S>=theta
    tau_curve = []
    for th in np.linspace(0.30, 0.95, 40):
        sch = {ft: sched_tau_aware(c, K, th) for ft, c in classes.items()}
        cost, meanF, minF, _ = eval_policy(classes, K, sch)
        tau_curve.append((cost, meanF, minF))
    # fixed-period: sweep uniform period p
    fix_curve = []
    for p in range(1, K + 1):
        sch = {ft: sched_fixed(p, K) for ft in classes}
        cost, meanF, minF, _ = eval_policy(classes, K, sch)
        fix_curve.append((cost, meanF, minF))
    # endpoints
    never = eval_policy(classes, K, {ft: set() for ft in classes})
    always = eval_policy(classes, K, {ft: set(range(1, K + 1)) for ft in classes})

    print(f"\nendpoints: never cost={never[0]} meanS={never[1]:.2f} minS={never[2]:.2f} | "
          f"always cost={always[0]} meanS={always[1]:.2f} minS={always[2]:.2f}")

    # headline 1: minimum cost each policy needs to GUARANTEE worst-case survival >= theta
    def min_cost_for(curve, th):
        ok = [c for c, _, mn in curve if mn >= th - 1e-9]
        return min(ok) if ok else None
    ct = min_cost_for(tau_curve, theta)
    cf = min_cost_for(fix_curve, theta)
    print(f"\n=== cost to GUARANTEE worst-case survival >= theta={theta} ===")
    print(f"  tau-aware:     {ct} re-injections")
    print(f"  fixed-period:  {cf} re-injections")
    if ct and cf:
        print(f"  -> tau-aware needs {100*(1-ct/cf):.0f}% fewer re-injections for the same guarantee")

    # headline 2: at matched cost (only WITHIN both policies' achieved ranges), worst-case survival
    print("\n=== matched-cost worst-case survival (within achieved ranges) ===")
    lo = max(min(c for c, _, _ in tau_curve), min(c for c, _, _ in fix_curve))
    hi = min(max(c for c, _, _ in tau_curve), max(c for c, _, _ in fix_curve))
    def interp(curve, cost):
        cs = np.array([c for c, _, _ in curve]); ms = np.array([m for _, _, m in curve])
        o = np.argsort(cs); return float(np.interp(cost, cs[o], ms[o]))
    wins = 0; tot = 0
    for target in [t for t in [40, 50, 55, 60, 65, 70] if lo <= t <= hi]:
        tw = interp(tau_curve, target); fw = interp(fix_curve, target)
        tot += 1; wins += (tw >= fw - 1e-6)
        print(f"  cost~{target:2d}: tau-aware minS={tw:.3f}  fixed-period minS={fw:.3f}  "
              f"-> {'tau-aware>=' if tw>=fw-1e-6 else 'fixed>'}")
    if tot:
        print(f"  tau-aware >= fixed-period at {wins}/{tot} overlapping cost budgets")

    out = {"K": K, "theta": theta, "classes": classes,
           "tau_curve": tau_curve, "fixed_curve": fix_curve,
           "never": {"cost": never[0], "meanS": never[1], "minS": never[2]},
           "always": {"cost": always[0], "meanS": always[1], "minS": always[2]}}
    os.makedirs("out/pilot", exist_ok=True)
    json.dump(out, open("out/pilot/scheduler_sim.json", "w"), indent=2)
    print("\nwrote out/pilot/scheduler_sim.json")


if __name__ == "__main__":
    main()
