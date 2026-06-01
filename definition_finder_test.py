"""
definition_finder_test.py
=========================
PASS/FAIL test harness for find_definition().

Each test case specifies:
  - string       : Full Swedish string.
  - word           : The surface-form word to look up.
  - char_index     : Character offset of (any character within) the target word.
  - expect_id_prefix   : The returned sense ID must start with this prefix.
  - expect_in_def      : This substring must appear in the returned definition.
  - expect_not_in_def  : This substring must NOT appear in the returned definition.
  (All "expect_*" fields are optional; omit any that are not relevant.)

Expected values are grounded in the actual Lexin API data confirmed by
a probe on 2026-05-26:

  banan..1  (fruit) : "en böjd tropisk frukt med kraftigt gult skal"
                       forms: bananen, bananer, bananerna
  bana..1   (track) : "väg; järnväg"            forms: banan, banor, banorna
  bana..2           : "färdväg"                  forms: banan, banor, banorna
  bana..3           : "karriär"                  forms: banan, banor, banorna
  bana..4           : "anläggning med plant underlag (särskilt för tävlingar)"
                       forms: banan, banor, banorna
  får..1    (sheep) : "ett ullhårigt djur som hålls som husdjur (släktet Ovis)"
  får..2    (verb)  : "tar emot, erhåller"
  får..3    (verb)  : "har tillåtelse att"
  får..4    (verb)  : "är tvungen, måste"
"""

import traceback
from definition_finder import find_definition

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "banan — frukt (obestämd sg, föregås av 'En')",
        "string":    "En banan låg på banan där bilarna körde.",
        "word":        "banan",
        "char_index":  3,
        # 'En' is an explicit indefinite article → SpaCy assigns Definite=Ind.
        # The fruit entry (banan..1) has 'banan' as its baseform (= indef sg).
        # All bana..x entries have 'banan' as their def.sg inflection → incompatible.
        "expect_id_prefix":  "lexin--banan",
        "expect_in_def":     "frukt",
    },
    {
        "name": "banan — bana (bestämd sg, föregås av 'på')",
        "string":    "En banan låg på banan där bilarna körde.",
        "word":        "banan",
        "char_index":  16,
        # 'på banan' has no article; SpaCy morphology → Definite=Def.
        # 'banan' is the def.sg form of bana..x → compatible.
        # The fruit entry (banan..1) has its def.sg as 'bananen', not 'banan'
        # → incompatible with a definite reading.
        "expect_id_prefix":  "lexin--bana",
        "expect_not_in_def": "frukt",
    },
    {
        "name": "får — verb",
        "string":    "Får får får? Nej, får får lamm!",
        "word":        "får",
        "char_index":  0,
        # "Får X Y?" — verb-initial question. SpaCy should tag as VERB.
        # POS filter retains only verb senses (får..2/3/4).
        # The sheep noun sense (får..1) must NOT be returned.
        "expect_in_def": "erhåller",
        "expect_not_in_def": "djur",
    },
    {
        "name": "får — substantiv",
        "string":    "Får får får? Nej, får får lamm!",
        "word":        "får",
        "char_index":  4,
        # Second token is the subject noun 'får' (sheep).
        # SpaCy should tag as NOUN; POS filter retains only får..1.
        "expect_id_prefix": "lexin--får..1",
        "expect_in_def":    "djur",
    },
    {
        "name": "Math problem",
        "string": "Förenkla bråket så långt som möjligt.",
        "word": "bråket",
        "char_index": 10,
        "expect_id_prefix": "lexin--bråk",
        "expect_in_def": "tal",
    },
    {
        "name":"Bära en bar",
        "string": "Han bar en bar bar in i en bar.",
        "word": "bar",
        "char_index": 11,
        "expect_id_prefix": "lexin--bar",
        "expect_in_def": "inte täckt"
    },
    {
    "name": "Rätten — domstol",
    "string": (
        "Rätten samlades klockan nio på morgonen. "
        "Domaren frågade om den åtalade förstod anklagelsen. "
        "Försvarsadvokaten begärde ordet."
    ),
    "word": "Rätten",
    "char_index": 0,
    "expect_id_prefix": "lexin--rätt",
    "expect_in_def": "domstol",
    },
    {
    "name": "Rätten — maträtt",
    "string": (
        "Rätten serverades med kokt potatis och sås. "
        "Kocken hade lagat maten sedan tidigt på morgonen."
    ),
    "word": "Rätten",
    "char_index": 0,
    "expect_in_def": "mat", 
    }
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _check(result: dict | None, case: dict) -> list[str]:
    """Return a list of failure messages; empty list = PASS."""
    failures = []

    if result is None:
        failures.append("find_definition returned None (no candidates)")
        return failures

    defn = result.get("definition", "")
    sid  = result.get("id", "")

    if prefix := case.get("expect_id_prefix"):
        if not sid.startswith(prefix):
            failures.append(f"ID mismatch: got '{sid}', expected prefix '{prefix}'")

    if substr := case.get("expect_in_def"):
        if substr not in defn:
            failures.append(f"Definition missing '{substr}': got '{defn}'")

    if substr := case.get("expect_not_in_def"):
        if substr in defn:
            failures.append(f"Definition wrongly contains '{substr}': got '{defn}'")

    return failures

import time
def run_tests() -> None:
    t = time.perf_counter()
    passed = 0
    failed = 0

    for case in TEST_CASES:
        name        = case["name"]
        string    = case["string"]
        word        = case["word"]
        char_index  = case["char_index"]

        print(f"\n" + "-" * 70)
        print(f"TEST : {name}")
        print(f"      '{string}'  word='{word}'  @{char_index}")

        try:
            result = find_definition(string, word, char_index)
        except Exception as exc:
            print(f"  EXCEPTION: {exc}")
            traceback.print_exc()
            failed += 1
            print("  ✗ FAIL")
            continue

        failures = _check(result, case)

        if result:
            print(f"  id         : {result['id']}")
            print(f"  definition : {result['definition']}")
            print(f"  score      : {result['score']:.4f}")

        if failures:
            for msg in failures:
                print(f"  ✗ {msg}")
            print("  ✗ FAIL")
            failed += 1
        else:
            print("  [PASS]")
            passed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{passed + failed} passed")
    print("=" * 70)
    print(f"Total execution time: {time.perf_counter() - t:.4f} seconds")


if __name__ == "__main__":
    run_tests()