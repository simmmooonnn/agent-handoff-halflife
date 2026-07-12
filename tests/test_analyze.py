import math, random
import numpy as np
from analyze import (fit_tau, fit_power_law, is_below_floor, moat_verdict,
                     bootstrap_survival_ci, bootstrap_tau_ci,
                     fit_linear, fit_constant, compare_decay_models)

def _synth(ks, tau, f=0.0, S0=1.0):
    return [f + (S0-f)*math.exp(-k/tau) for k in ks]

def test_fit_recovers_known_tau():
    ks = [0,1,2,4,8,16]
    S = _synth(ks, tau=3.0)
    out = fit_tau(ks, S, chance=0.0)
    assert abs(out["tau"] - 3.0) < 0.3
    assert out["r2"] > 0.99

def test_power_law_recovers_alpha():
    sizes = [1e8, 1e9, 1e10, 1e11]
    taus = [s**0.5 / 1e4 for s in sizes]
    out = fit_power_law(sizes, taus)
    assert abs(out["alpha"] - 0.5) < 0.05

def test_below_floor_flag():
    assert is_below_floor(S0=0.36, chance=0.33, margin=0.15) is True
    assert is_below_floor(S0=0.76, chance=0.33, margin=0.15) is False


def _make_rows(condition, ks, tau, f=0.0, S0=1.0, n_per_k=40, rng=None):
    """Build synthetic rows where each (condition, k) has n_per_k Bernoulli trials."""
    if rng is None:
        rng = random.Random(42)
    rows = []
    for k in ks:
        p = f + (S0 - f) * math.exp(-k / tau)
        p = max(0.0, min(1.0, p))
        for _ in range(n_per_k):
            rows.append({"condition": condition, "k": k,
                         "correct": int(rng.random() < p), "ftype": "test"})
    return rows


def test_moat_passes_when_handoff_decays_and_longctx_flat():
    """IDEAL moat: handoff decays (tau≈3), longctx stays flat-high → passes True.
    (A flat long-context curve means the fact is preserved — the result we want.)"""
    rng = random.Random(13)
    ks = [0, 1, 2, 4, 8, 16]
    rows = _make_rows("handoff", ks, tau=3.0, S0=1.0, f=0.0, n_per_k=80, rng=rng)
    for k in ks:                       # longctx flat-high at 1.0 (fact preserved)
        for _ in range(80):
            rows.append({"condition": "longctx", "k": k, "correct": 1, "ftype": "test"})
    result = moat_verdict(rows, chance=0.0)
    assert result["passes"] is True, f"flat-high longctx is the ideal moat: {result}"


def test_moat_fails_when_handoff_does_not_decay():
    """Both arms flat → handoff doesn't decay → passes False (the real spurious case)."""
    ks = [0, 1, 2, 4, 8, 16]
    rows = []
    for cond in ("handoff", "longctx"):
        for k in ks:
            for _ in range(80):
                rows.append({"condition": cond, "k": k, "correct": 1, "ftype": "test"})
    result = moat_verdict(rows, chance=0.0)
    assert result["passes"] is False, f"no handoff decay must not pass: {result}"


def test_moat_fails_when_both_decay_equally():
    """handoff and longctx decay the same (tau≈3 each) → handoff not special → passes False."""
    rng = random.Random(13)
    ks = [0, 1, 2, 4, 8, 16]
    rows = _make_rows("handoff", ks, tau=3.0, S0=1.0, f=0.0, n_per_k=80, rng=rng)
    rows += _make_rows("longctx", ks, tau=3.0, S0=1.0, f=0.0, n_per_k=80, rng=rng)
    result = moat_verdict(rows, chance=0.0)
    assert result["passes"] is False, f"equal decay must not pass: {result}"


def _make_rows_with_facts(condition, ks, tau, f=0.0, S0=1.0, n_facts=20, rng=None):
    """Build synthetic rows with fact_id, n_facts facts each evaluated at all ks."""
    if rng is None:
        rng = random.Random(42)
    rows = []
    for fid in range(n_facts):
        for k in ks:
            p = f + (S0 - f) * math.exp(-k / tau)
            p = max(0.0, min(1.0, p))
            rows.append({"condition": condition, "k": k,
                         "correct": int(rng.random() < p),
                         "fact_id": f"f{fid}"})
    return rows


def test_bootstrap_survival_ci_brackets_mean():
    ks = [0, 1, 2, 4, 8]
    rows_small = _make_rows_with_facts("handoff", ks, tau=4.0, n_facts=10, rng=random.Random(1))
    rows_large = _make_rows_with_facts("handoff", ks, tau=4.0, n_facts=80, rng=random.Random(1))
    ci_small = bootstrap_survival_ci(rows_small, "handoff", n_boot=500, rng_seed=0)
    ci_large = bootstrap_survival_ci(rows_large, "handoff", n_boot=500, rng_seed=0)
    for k in ks:
        s = ci_small[k]
        l = ci_large[k]
        # CI must bracket the observed mean
        assert s["lo"] <= s["mean"] <= s["hi"], f"small CI does not bracket mean at k={k}: {s}"
        assert l["lo"] <= l["mean"] <= l["hi"], f"large CI does not bracket mean at k={k}: {l}"
        # More facts → narrower CI
        assert (l["hi"] - l["lo"]) <= (s["hi"] - s["lo"]) + 0.05, (
            f"larger sample should have narrower or equal CI at k={k}: small={s['hi']-s['lo']:.3f} large={l['hi']-l['lo']:.3f}")


def test_bootstrap_tau_ci_contains_true_tau():
    ks = [0, 1, 2, 4, 8, 16]
    true_tau = 5.0
    rows = _make_rows_with_facts("handoff", ks, tau=true_tau, n_facts=40, rng=random.Random(7))
    result = bootstrap_tau_ci(rows, "handoff", chance=0.0, n_boot=300, rng_seed=42)
    assert result["lo"] <= result["tau"] <= result["hi"], "median not inside CI"
    assert result["lo"] <= true_tau <= result["hi"], (
        f"true tau {true_tau} not in CI [{result['lo']:.2f}, {result['hi']:.2f}]")


# --- decay-shape model-comparison tests ---

def test_compare_picks_exponential_on_exp_data():
    ks = [0, 1, 2, 4, 8, 16]
    tau = 2.0
    S = [math.exp(-k / tau) for k in ks]
    result = compare_decay_models(ks, S, chance=0.0)
    assert result["best"] == "exponential", (
        f"Expected 'exponential', got '{result['best']}'; AICs={result['models']}")


def test_compare_picks_linear_on_linear_data():
    ks = [0, 1, 2, 4, 8, 16]
    S = [max(0.0, 1.0 - 0.05 * k) for k in ks]
    result = compare_decay_models(ks, S, chance=0.0)
    assert result["best"] == "linear", (
        f"Expected 'linear', got '{result['best']}'; AICs={result['models']}")


def test_fit_linear_recovers_slope():
    ks = [0, 1, 2, 4, 8, 16]
    S = [2.0 + 3.0 * k for k in ks]
    out = fit_linear(ks, S)
    assert abs(out["a"] - 2.0) < 1e-6, f"intercept wrong: {out['a']}"
    assert abs(out["b"] - 3.0) < 1e-6, f"slope wrong: {out['b']}"
    assert abs(out["r2"] - 1.0) < 1e-6, f"r2 wrong: {out['r2']}"
