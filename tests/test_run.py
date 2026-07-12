import hashlib
from facts import make_facts
from backends import MockBackend
from run import run_sweep
from analyze import survival_curve, fit_tau, compare_decay_models

def test_run_sweep_shape():
    # mock agent that always preserves the fact -> survival should be ~1 everywhere
    b = MockBackend("mock", lambda p: next((l for l in p.splitlines()
                    if l.startswith("CARRY:")), "CARRY:"))
    facts = make_facts(5, 0)
    rows = run_sweep(b, facts, ks=[0,1,2], conditions=["handoff","longctx"],
                     budget=400, load="light", filler="x y z", seeds=[0], level="actionable")
    assert len(rows) == 5*3*2*1
    assert {r["condition"] for r in rows} == {"handoff","longctx"}
    assert all(isinstance(r["correct"], bool) for r in rows)

def test_lossy_mock_pipeline_recovers_decay():
    # End-to-end pipeline check (NOT a real-model claim): a stochastic-but-deterministic
    # mock drops the carried content at each hop with prob ~0.3 (keyed by the per-hop
    # cache item_id), so aggregate survival should decay ~0.7^k. The analysis pipeline
    # must recover a finite tau and NOT call the curve constant.
    def lossy(prompt, item_id, max_tokens):
        carry = next((l for l in prompt.splitlines() if l.startswith("CARRY:")), "CARRY:")
        if ":h" in item_id:  # a hop summarization call: keep ~70% of the time
            h = int(hashlib.sha1(item_id.encode()).hexdigest(), 16) % 100
            return carry if h < 70 else "CARRY: (lost)"
        return carry  # final query echoes whatever survived

    class LossyBackend:
        name = "lossy"
        def generate(self, prompt, item_id, max_tokens):
            return lossy(prompt, item_id, max_tokens)

    facts = make_facts(40, 0)
    rows = run_sweep(LossyBackend(), facts, ks=[0,1,2,4,8], conditions=["handoff"],
                     budget=20, load="none", filler="x", seeds=[0], level="actionable")
    ks, S = survival_curve(rows, "handoff")
    assert S[0] >= 0.85 and S[-1] < S[0] - 0.2    # decays clearly from a high start
    fit = fit_tau(ks, S, chance=0.0)
    assert 0.5 < fit["tau"] < 20 and fit["r2"] > 0.8   # finite, well-fit tau
    assert compare_decay_models(ks, S, chance=0.0)["best"] != "constant"
