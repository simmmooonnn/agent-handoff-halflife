from facts import make_facts
from backends import MockBackend
from relay import run_chain, item_id_for

# A mock "agent": echoes the fact statement if it still sees it, truncated to budget.
def make_mock():
    def fn(prompt):
        # crude: return the last 'STATEMENT:' line if present (simulates lossy carry)
        for line in prompt.splitlines():
            if line.startswith("CARRY:"):
                return line[:9999]
        return "CARRY: (nothing)"
    return MockBackend("mock", fn)

def test_item_id_stable_and_condition_sensitive():
    f = make_facts(5, 0)[0]
    a = item_id_for(f, 4, "handoff", 200, "light", 0)
    b = item_id_for(f, 4, "handoff", 200, "light", 0)
    c = item_id_for(f, 4, "longctx", 200, "light", 0)
    assert a == b and a != c

def test_all_conditions_run_and_return_str():
    f = make_facts(5, 0)[0]
    b = make_mock()
    for cond in ("handoff", "longctx", "verbatim", "nofact"):
        out = run_chain(b, f, k=2, condition=cond, budget=200, load="light",
                        filler="lorem ipsum dolor sit amet", seed=0)
        assert isinstance(out, str) and len(out) > 0

def test_verbatim_survival_independent_of_k():
    # Verbatim does no model rewrite: carry is always fact.statement regardless of k.
    # The final answer must be identical for k=1 and k=8.
    f = make_facts(5, 0)[0]
    b = make_mock()
    out_k1 = run_chain(b, f, k=1, condition="verbatim", budget=200, load="none",
                       filler="lorem ipsum", seed=0)
    out_k8 = run_chain(b, f, k=8, condition="verbatim", budget=200, load="none",
                       filler="lorem ipsum", seed=0)
    assert out_k1 == out_k8, (
        f"verbatim answer must not change with k: k=1 gave {out_k1!r}, k=8 gave {out_k8!r}")

def test_budget_communicated_as_word_limit():
    # Budget is a generation-time WORD constraint (model chooses what to keep),
    # NOT a post-hoc hard char truncation. Assert the hop prompt tells the model the limit.
    seen = []
    b = MockBackend("mock", lambda p: (seen.append(p) or "CARRY: ok"))
    f = make_facts(5, 0)[0]
    run_chain(b, f, k=2, condition="handoff", budget=25, load="none", filler="f", seed=0)
    hop_prompts = [p for p in seen if "Summarize" in p]
    assert hop_prompts, "expected hop summarization prompts"
    assert all("at most 25 words" in p for p in hop_prompts)
