from facts import FACT_TYPES, make_facts, chance_level

def test_fact_types_cover_taxonomy():
    assert set(FACT_TYPES) == {"numeric", "entity", "negation", "preference"}

def test_same_type_facts_have_distinct_subjects():
    # Within a type, statements must differ by SUBJECT (not just value), so a same-type
    # distractor set keeps the probe identifiable by its query. Check the first 8 numeric.
    nums = [f for f in make_facts(40, 0) if f.ftype == "numeric"][:8]
    assert len({f.query for f in nums}) == 8, "numeric probes must have distinct subjects/queries"

def test_make_facts_deterministic_and_typed():
    a = make_facts(20, seed=0)
    b = make_facts(20, seed=0)
    assert [f.statement for f in a] == [f.statement for f in b]   # reproducible
    assert len(a) == 20
    assert all(f.ftype in FACT_TYPES for f in a)
    assert all(f.answer and f.query for f in a)
    assert len({f.fact_id for f in a}) == 20                       # unique ids

def test_chance_level_known_values():
    assert chance_level("numeric") < 0.05      # open numeric answer ~ near 0 chance
    assert 0.0 < chance_level("preference") <= 0.5
