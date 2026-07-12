# CI BACKFILL (top-conf hardening item #3). Cache-only replay: reconstruct PER-FACT
# outcomes for the paper's headline contrasts from the existing response caches (zero API
# calls; a dummy key satisfies the presence check), then bootstrap 95% percentile CIs
# over facts (the statistical unit; seeds are not replication under temp-0 greedy).
import os, sys
os.environ.setdefault("ANTHROPIC_API_KEY", "cache-only-replay")
import numpy as np
from statistics import mean
from facts import make_facts, FACT_TYPES
from grade import grade
from relay import run_chain
from run import FILLER, _build_backend

RNG = np.random.default_rng(0)
B_REPS = 2000


def ci(vals):
    v = np.asarray(vals, float)
    bs = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(B_REPS)]
    return round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), \
        round(float(np.percentile(bs, 97.5)), 3)


def diff_ci(a, b):
    """paired bootstrap over facts of mean(a)-mean(b); a,b aligned per-fact lists"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    bs = [d[RNG.integers(0, len(d), len(d))].mean() for _ in range(B_REPS)]
    return round(float(d.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), \
        round(float(np.percentile(bs, 97.5)), 3)


def cell_perfact(backend, pset, B, k):
    out = []
    for probe, dist in pset:
        r = run_chain(backend, probe, k, "handoff", budget=B, load="light",
                      filler=FILLER, seed=0, distractors=dist)
        out.append(1 if grade(probe, r, "actionable") else 0)
    return out


def main():
    from e12_provisioning import probe_set
    pset32 = probe_set()  # PROV_PROBES default 8/type = 32, matches e12/e13 runs

    print("=== CI BACKFILL (cache-only replay, bootstrap over facts, 95% CI) ===\n")

    # ---- e8 (Sonnet K=6): slots_copy vs neutral, per type ----
    import e8_structured_slots as e8
    e8.K, e8.PROBES = 6, 12
    bk8 = _build_backend("anthropic", "claude-sonnet-4-6", "data/cache_slots_claude-sonnet-4-6.json")
    facts = make_facts(400, seed=1)
    bt = {}
    for f in facts:
        bt.setdefault(f.ftype, []).append(f)
    print("[e8 Sonnet K=6] slots_copy - neutral (paired over facts):")
    all_neu, all_cop = [], []
    for t in FACT_TYPES:
        probes = bt[t][:12]
        pool = [f for tt in FACT_TYPES for f in bt[tt][12:12 + 3]]
        neu, cop = [], []
        for probe in probes:
            dist = tuple(d for d in pool if d.fact_id != probe.fact_id)[:7]
            neu.append(1 if grade(probe, run_chain(bk8, probe, 6, "handoff", budget=25,
                       load="light", filler=FILLER, seed=0, distractors=dist), "actionable") else 0)
            s, _ = e8.run_slots(bk8, probe, dist, 0, "copy")
            cop.append(s)
        m, lo, hi = diff_ci(cop, neu)
        print(f"  {t:>11}: +{m:.3f} [{lo:+.3f}, {hi:+.3f}]{'  CI excl 0' if lo > 0 else ''}")
        all_neu += neu; all_cop += cop
    m, lo, hi = diff_ci(all_cop, all_neu)
    print(f"  {'ALL':>11}: +{m:.3f} [{lo:+.3f}, {hi:+.3f}]{'  CI excl 0' if lo > 0 else ''}\n")

    # ---- e8b: copy - regen gap at K=8 and K=20 ----
    print("[e8b Sonnet] slots_copy - slots_regen:")
    for K in (8, 20):
        e8.K = K
        gapc, gapr = [], []
        for t in FACT_TYPES:
            probes = bt[t][:10]
            pool = [f for tt in FACT_TYPES for f in bt[tt][10:10 + 3]]
            for probe in probes:
                dist = tuple(d for d in pool if d.fact_id != probe.fact_id)[:7]
                sc, _ = e8.run_slots(bk8, probe, dist, 0, "copy")
                sr, _ = e8.run_slots(bk8, probe, dist, 0, "regen")
                gapc.append(sc); gapr.append(sr)
        m, lo, hi = diff_ci(gapc, gapr)
        print(f"  K={K:>2}: +{m:.3f} [{lo:+.3f}, {hi:+.3f}]{'  CI excl 0' if lo > 0 else ''}")
    print()

    # ---- e9 (numeric): mem0 vs free_text; ours vs free_text at k=8 ----
    from e9_sota_showdown import run_mem0
    bk9 = _build_backend("anthropic", "claude-sonnet-4-6", "data/cache_showdown_claude-sonnet-4-6.json")
    probes = bt["numeric"][:12]
    pool = [f for tt in FACT_TYPES for f in bt[tt][12:12 + 3]]
    print("[e9 Sonnet numeric k=8]:")
    ftv, m0v, slv = [], [], []
    e8.K = 8
    for probe in probes:
        dist = tuple(d for d in pool if d.fact_id != probe.fact_id)[:7]
        ftv.append(1 if grade(probe, run_chain(bk9, probe, 8, "handoff", budget=25,
                   load="light", filler=FILLER, seed=0, distractors=dist), "actionable") else 0)
        m0v.append(run_mem0(bk9, probe, dist, 8, 0))
        sc, _ = e8.run_slots(bk9, probe, dist, 0, "copy")
        slv.append(sc)
    for name, v in (("free_text", ftv), ("mem0", m0v), ("ours_slot", slv)):
        m, lo, hi = ci(v)
        print(f"  {name:>10}: {m:.3f} [{lo:.3f}, {hi:.3f}]")
    m, lo, hi = diff_ci(m0v, ftv)
    print(f"  mem0-free_text: +{m:.3f} [{lo:+.3f}, {hi:+.3f}]{'  CI excl 0' if lo > 0 else ''}\n")

    # ---- e13 Haiku: budget gain at k=4 (B=100 vs B=15) ----
    bk13 = _build_backend("anthropic", "claude-haiku-4-5", "data/cache_sat_claude-haiku-4-5.json")
    lo15 = cell_perfact(bk13, pset32, 15, 4)
    hi100 = cell_perfact(bk13, pset32, 100, 4)
    m, lo, hi = diff_ci(hi100, lo15)
    print(f"[e13 Haiku k=4] S(B=100)-S(B=15) gain: +{m:.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"{'  CI excl 0 -> BINDS significant' if lo > 0 else ''}")

    # ---- e12 Sonnet: same gain (B=100 provisioning cell vs B=15) ----
    bk12 = _build_backend("anthropic", "claude-sonnet-4-6", "data/cache_prov_claude-sonnet-4-6.json")
    s15 = cell_perfact(bk12, pset32, 15, 4)
    s100 = cell_perfact(bk12, pset32, 100, 4)
    m, lo, hi = diff_ci(s100, s15)
    sat = "  CI excl 0" if lo > 0 else "  CI INCLUDES 0 -> saturation"
    print(f"[e12 Sonnet k=4] S(B=100)-S(B=15) gain: +{m:.3f} [{lo:+.3f}, {hi:+.3f}]{sat}")


if __name__ == "__main__":
    main()
