import random
from dataclasses import dataclass

# 'relational' dropped: empirical no-fact chance ~0.875 (model guesses it) → not a memory probe.
FACT_TYPES = ["numeric", "entity", "negation", "preference"]

@dataclass
class Fact:
    fact_id: str
    ftype: str
    statement: str   # planted into step-0 context
    query: str       # asked at the end (names the SUBJECT so the probe is identifiable)
    answer: str      # canonical answer for grading

# Each type has a pool of DISTINCT subjects so same-type distractors compete for budget
# without making the probe ambiguous (the query names the subject).
_NUM_SUBJECTS = ["budget ceiling", "retry limit", "request timeout", "cache size",
                 "batch size", "port number", "max connections", "rate limit",
                 "session length", "thread count", "page size", "token budget"]
_ENT_ROLES = ["client contact", "project lead", "code reviewer", "account manager",
              "on-call engineer", "data steward", "release owner", "security officer",
              "product manager", "QA lead", "design lead", "support agent"]
_NAMES = ["Voss", "Quill", "Marlowe", "Okonkwo", "Dubois", "Nakamura",
          "Aaltonen", "Reyes", "Bianchi", "Haddad", "Sokolov", "Tan"]
_NEG_ITEMS = ["the staging server", "the export module", "the billing job",
              "the cache layer", "the auth service", "the search index",
              "the payment gateway", "the audit log", "the user database",
              "the message queue", "the config store", "the backup script"]
_PREF = [("indentation", ("tabs", "spaces", "either")),
         ("line length", ("80", "100", "120")),
         ("quote style", ("single", "double", "either")),
         ("test framework", ("pytest", "unittest", "nose")),
         ("branch naming", ("kebab-case", "snake_case", "slash/style")),
         ("logging level", ("debug", "info", "warning")),
         ("date format", ("ISO-8601", "US", "European")),
         ("import ordering", ("alphabetical", "grouped", "by-length")),
         ("commit style", ("imperative", "past-tense", "conventional")),
         ("doc style", ("google", "numpy", "rest"))]

def chance_level(ftype: str) -> float:
    # Coarse fallback only; pilots/analysis use the empirical 'nofact' baseline instead.
    return {"numeric": 0.001, "entity": 1/len(_NAMES), "negation": 0.5, "preference": 1/3}[ftype]

def make_facts(n: int, seed: int) -> list[Fact]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        ft = FACT_TYPES[i % len(FACT_TYPES)]
        j = i // len(FACT_TYPES)          # per-type index → distinct subject within a type
        if ft == "numeric":
            subj = _NUM_SUBJECTS[j % len(_NUM_SUBJECTS)]
            v = rng.randint(1000, 9999)
            f = Fact(f"f{i}", ft, f"The {subj} is {v}.",
                     f"What is the {subj}? Reply with the number only.", str(v))
        elif ft == "entity":
            role = _ENT_ROLES[j % len(_ENT_ROLES)]
            name = _NAMES[j % len(_NAMES)]
            f = Fact(f"f{i}", ft, f"The {role} is {name}.",
                     f"Who is the {role}? Reply with the name only.", name)
        elif ft == "negation":
            item = _NEG_ITEMS[j % len(_NEG_ITEMS)]
            forbidden = (j % 2 == 0)
            if forbidden:
                f = Fact(f"f{i}", ft, f"Under no circumstances modify {item}.",
                         f"Is it permitted to modify {item}? Answer yes or no.", "no")
            else:
                f = Fact(f"f{i}", ft, f"You may freely modify {item}.",
                         f"Is it permitted to modify {item}? Answer yes or no.", "yes")
        else:  # preference
            aspect, choices = _PREF[j % len(_PREF)]
            choice = choices[rng.randrange(len(choices))]
            f = Fact(f"f{i}", ft, f"For {aspect}, the user prefers {choice}.",
                     f"What does the user prefer for {aspect}? Reply with one word.", choice)
        out.append(f)
    return out
