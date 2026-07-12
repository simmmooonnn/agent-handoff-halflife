import re
from facts import Fact

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _wb(s: str) -> str:
    # word-boundary normalization: alphanumeric tokens joined by single spaces,
    # with sentinel spaces so containment tests align on whole tokens only
    return " " + " ".join(re.findall(r"[a-z0-9]+", s.lower())) + " "

def grade(fact: Fact, response: str, level: str, judge=None, boundary=True) -> bool:
    # boundary=True (default, R1.2 fix): the answer must appear as a whole-token
    # sequence. The legacy path (boundary=False) stripped ALL non-alphanumerics
    # including spaces before substring search, so "tan" matched inside
    # "important" -- kept only for the regrade sensitivity comparison.
    ans = fact.answer
    if level == "verbatim":
        return ans in response
    r = response.lower()
    if fact.ftype == "negation":
        first = next((w for w in re.findall(r"[a-z]+", r) if w in ("yes", "no")), None)
        return first == ans
    if level == "semantic" and judge is not None:
        return bool(judge(fact, response))
    if boundary:
        return _wb(ans) in _wb(response)
    return _norm(ans) in _norm(response)
