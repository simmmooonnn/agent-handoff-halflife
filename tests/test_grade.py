from facts import make_facts
from grade import grade

def test_verbatim_numeric():
    f = [x for x in make_facts(5, 0) if x.ftype == "numeric"][0]
    assert grade(f, f"The answer is {f.answer}.", "verbatim") is True
    assert grade(f, "The answer is 0000.", "verbatim") is False

def test_actionable_negation_is_lenient_on_form():
    # make_facts(20, 0) now has both polarities: some negation facts answer "no", some "yes"
    all_neg = [x for x in make_facts(20, 0) if x.ftype == "negation"]
    f_no  = next(x for x in all_neg if x.answer == "no")
    f_yes = next(x for x in all_neg if x.answer == "yes")
    # "no" polarity: forbidden item
    assert grade(f_no, "No, you must not.", "actionable") is True
    assert grade(f_no, "Yes, go ahead.", "actionable") is False
    # "yes" polarity: permitted item
    assert grade(f_yes, "Yes, that is fine.", "actionable") is True
    assert grade(f_yes, "No, do not do that.", "actionable") is False

def test_entity_case_insensitive_actionable():
    f = [x for x in make_facts(5, 0) if x.ftype == "entity"][0]
    assert grade(f, f"It is {f.answer.lower()}.", "actionable") is True
