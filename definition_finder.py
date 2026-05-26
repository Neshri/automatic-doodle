"""
definition_finder.py
====================
Swedish Word Sense Disambiguation using the Lexin dictionary (via the Karp API)
and BGE-M3 embeddings (via Ollama) as a fallback scorer.

Public interface
----------------
    find_definition(sentence, word, char_index) -> dict | None

The function is completely general — no hardcoded word-specific overrides.
"""

import spacy
import requests
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

nlp = spacy.load("sv_core_news_lg")

KARP_API   = "https://spraakbanken4.it.gu.se/karp/v7/query/lexin"
OLLAMA_API = "http://localhost:11434/api/embed"
EMBED_MODEL = "bge-m3"

# SpaCy Universal POS → Lexin POS code prefixes
_POS_MAP: dict[str, list[str]] = {
    "NOUN":  ["nn"],
    "PROPN": ["nn"],
    "VERB":  ["vb", "vbm"],
    "AUX":   ["vb", "vbm"],
    "ADJ":   ["jj", "hjj"],
    "ADV":   ["ab"],
    "PRON":  ["pn"],
    "NUM":   ["rg", "ro"],
    "INTJ":  ["in"],
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_embedding(texts: list[str]) -> np.ndarray:
    """
    Call the local Ollama embedding API forcing CPU usage.
    """
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        # Adding options here forces Ollama to offload 0 layers to GPU
        "options": {
            "num_gpu": 0
        }
    }
    
    r = requests.post(OLLAMA_API, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    if "embeddings" in data:
        return np.array(data["embeddings"])
    if "embedding" in data:
        return np.array([data["embedding"]])
        
    raise KeyError(
        f"Unexpected Ollama response — expected 'embeddings' key, got: {list(data.keys())}"
    )

def _fetch_candidates(word_form: str, spacy_lemma: str) -> list[dict]:
    """
    Fetch every Lexin sense where *word_form* appears as a baseform
    OR as an inflected form in the inflection table.

    Also queries *spacy_lemma* as a baseform when it differs from
    *word_form*, which handles cases where SpaCy correctly lemmatises an
    inflected token (e.g. "körde" → "köra").

    No hardcoded word-specific overrides — the API queries are fully general.

    Parameters
    ----------
    word_form   : Surface form from the sentence (lowercased here).
    spacy_lemma : Lemma as assigned by SpaCy (lowercased here).

    Returns
    -------
    List of candidate dicts with keys:
        id, pos, baseform, definition, inflections
    """
    wf    = word_form.lower()
    lemma = spacy_lemma.lower()

    # Query 1 — word_form is a baseform OR appears in the inflection table.
    # The sub-query syntax `inflectionTable(...)` searches inside the array.
    q_form = (
        f"or(equals|languages.baseform|{wf}"
        f"||inflectionTable(equals|writtenForm|{wf}))"
    )
    queries: set[str] = {q_form}

    # Query 2 — SpaCy's lemma may reveal additional senses not visible from
    # the surface form alone (e.g. lemma "bana" when the surface form is "banan").
    if lemma != wf:
        queries.add(f"equals|languages.baseform|{lemma}")

    seen_ids: set[str] = set()
    results: list[dict] = []

    for q in queries:
        resp = requests.get(KARP_API, params={"q": q, "size": 25})
        resp.raise_for_status()
        for hit in resp.json().get("hits", []):
            entry  = hit.get("entry", {})
            sense  = entry.get("sense", {})
            sid    = sense.get("senseid", "")
            defn   = sense.get("definition", {}).get("text", "")

            if not sid or not defn or sid in seen_ids:
                continue
            seen_ids.add(sid)

            swe = next(
                (l for l in entry.get("languages", []) if l.get("lang") == "swe"),
                {},
            )

            # The Karp API occasionally returns `baseform` as a list of strings
            # rather than a plain string.  Normalise to str so that all
            # downstream callers can safely call .lower() on it.
            raw_baseform = swe.get("baseform", wf)
            if isinstance(raw_baseform, list):
                raw_baseform = raw_baseform[0] if raw_baseform else wf

            results.append(
                {
                    "id":          sid,
                    "pos":         swe.get("partOfSpeech", "?"),
                    "baseform":    raw_baseform,
                    "definition":  defn,
                    "inflections": entry.get("inflectionTable", []),
                }
            )

    return results


# ---------------------------------------------------------------------------
# Filtering pipeline
# ---------------------------------------------------------------------------

def _filter_by_pos(candidates: list[dict], spacy_pos: str) -> list[dict]:
    """
    Hard POS filter: keep only candidates whose Lexin POS is compatible
    with SpaCy's Universal POS tag.

    Lexin entries with POS '?' (phrasal/idiomatic entries without a
    grammatical category) are always removed.
    """
    allowed = _POS_MAP.get(spacy_pos, [])
    if not allowed:
        # Unknown POS — keep all rather than silently discard everything.
        return [c for c in candidates if c["pos"] != "?"]
    return [
        c for c in candidates
        if c["pos"] != "?" and any(p in c["pos"] for p in allowed)
    ]


def _form_role(candidate: dict, word_form: str) -> str | None:
    """
    Determine what grammatical role *word_form* plays in *candidate*.

    Returns
    -------
    "baseform"  — word_form equals the Lexin baseform (sg.indef for nouns).
    "Def"       — word_form appears in the inflection table as a definite form.
    "Ind"       — word_form appears in the inflection table as an indefinite form.
    "other"     — word_form is in the table but definiteness is not parseable
                  from the MSD string (e.g. verb present-tense forms).
    None        — word_form is not found in this candidate at all.
    """
    wf = word_form.lower()

    if candidate["baseform"].lower() == wf:
        return "baseform"

    for infl in candidate.get("inflections", []):
        if infl.get("writtenForm", "").lower() != wf:
            continue

        # Parse MSD string.  Check "indef"/"obest" BEFORE "def"/"best"
        # because "indef" contains "def" as a substring — checking "def"
        # first would incorrectly classify indefinite forms as definite.
        msd = (infl.get("lexinMsd", "") + " " + infl.get("msd", "")).lower()

        if "indef" in msd or "obest" in msd:
            return "Ind"
        if "best" in msd or ".def" in msd:
            return "Def"

        # Form found but MSD is not definiteness-bearing (e.g. verb forms).
        return "other"

    return None  # Word form not found anywhere in this candidate.


def _filter_by_form(candidates: list[dict], word_form: str) -> list[dict]:
    """
    Hard form-existence filter: remove candidates that have a non-empty
    inflection table that does NOT contain *word_form* at all — meaning
    the surface word form is morphologically impossible for that sense.

    Candidates without any inflection data pass through unchanged (we
    cannot falsify them).
    """
    def _ok(c: dict) -> bool:
        if not c.get("inflections"):
            return True  # No inflection table → no evidence to reject.
        return _form_role(c, word_form) is not None

    return [c for c in candidates if _ok(c)]


def _get_expected_definiteness(token) -> str | None:
    """
    Infer the expected definiteness of *token* from SpaCy's dependency
    parse and morphological tags.

    Priority
    --------
    1. Explicit determiner child (article, quantifier, numeral modifier).
    2. The token's own morphological Definite feature as a fallback.

    Returns 'Def', 'Ind', or None.
    """
    for child in token.children:
        if child.dep_ in ("det", "quant", "nummod"):
            morph = child.morph.to_dict()
            if "Definite" in morph:
                return morph["Definite"]
    return token.morph.to_dict().get("Definite")


def _filter_by_definiteness(
    candidates: list[dict],
    word_form: str,
    expected_def: str,
) -> list[dict]:
    """
    Definiteness filter for nouns: keep only candidates where *word_form*
    occurs in the role that matches *expected_def*.

    'baseform' role is treated as indefinite singular (Lexin convention).

    If the filter would remove all candidates (e.g. MSD data is absent),
    the original list is returned unchanged — this function never produces
    an empty set.
    """
    def _compatible(c: dict) -> bool:
        role = _form_role(c, word_form)
        if role is None or role == "other":
            return True  # Cannot falsify → keep.
        if role == "baseform":
            return expected_def == "Ind"
        return role == expected_def  # "Def" or "Ind"

    filtered = [c for c in candidates if _compatible(c)]
    return filtered if filtered else candidates  # Safety net.


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def _lesk_score(candidate: dict, doc, target_token) -> int:
    """
    Classic Lesk word-overlap score: count how many lemmas of the sentence's
    content words also appear (as raw tokens) in the definition text.

    Uses SpaCy lemmas for the context side to collapse inflected forms;
    uses simple whitespace tokenisation on the definition side (since we
    do not parse the definition text with SpaCy).
    """
    context_lemmas = {
        t.lemma_.lower()
        for t in doc
        if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV")
        and t.i != target_token.i
        and not t.is_stop
        and len(t.lemma_) > 2
    }
    def_tokens = set(candidate["definition"].lower().split())
    return len(context_lemmas & def_tokens)


def _embedding_scores(
    candidates: list[dict],
    sentence: str,
    target_token,
    doc,
) -> np.ndarray:
    """
    Asymmetric BGE-M3 cosine-similarity scores.

    The query is a cloze prompt that foregrounds the surrounding context
    without mentioning the target word, biasing the embedding toward the
    semantics of the surrounding tokens.  The candidate documents are the
    plain definition strings.

    Raises
    ------
    Same exceptions as _get_embedding() — callers must handle Ollama errors.
    """
    # Blank out the target word in the sentence.
    blanked = (
        sentence[: target_token.idx]
        + "___"
        + sentence[target_token.idx + len(target_token.text) :]
    )

    # Extract surrounding content-word lemmas as an explicit semantic hint.
    content_lemmas = [
        t.lemma_
        for t in doc
        if t.pos_ in ("NOUN", "VERB", "ADJ")
        and t.i != target_token.i
        and not t.is_stop
        and len(t.lemma_) > 2
    ]
    hint = ", ".join(content_lemmas) if content_lemmas else "okänd kontext"

    query = (
        f"Meningen '{blanked}' (nyckelord: {hint}). "
        f"Det obekanta ordet ___ syftar på:"
    )

    all_texts = [query] + [c["definition"] for c in candidates]
    embeds    = _get_embedding(all_texts)  # Raises on any Ollama failure.

    q_vec  = embeds[0]
    d_vecs = embeds[1:]
    norms  = np.linalg.norm(d_vecs, axis=1) * np.linalg.norm(q_vec)
    norms  = np.where(norms == 0, 1e-10, norms)
    return np.dot(d_vecs, q_vec) / norms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_definition(
    sentence: str,
    word: str,
    char_index: int,
) -> dict | None:
    """
    Find the correct Lexin definition for *word* at *char_index* in *sentence*.

    Parameters
    ----------
    sentence   : Full Swedish sentence.
    word       : The surface-form word to disambiguate.
    char_index : Character offset of any character within the word.

    Returns
    -------
    A dict with keys:
        "definition" : str   — The winning Lexin definition text.
        "id"         : str   — The Lexin sense ID.
        "score"      : float — Numeric confidence (higher = more certain).
    or None if no candidates were found after filtering.

    Raises
    ------
    ValueError            : No SpaCy token at char_index.
    requests.HTTPError    : Any HTTP error from Karp or Ollama.
    requests.ConnectionError : Ollama or Karp is unreachable.
    KeyError              : Unexpected Ollama response format.
    """
    # ── Step 1: Tokenise with SpaCy ──────────────────────────────────────
    doc   = nlp(sentence)
    token = next(
        (t for t in doc if t.idx <= char_index < t.idx + len(t.text)),
        None,
    )
    if token is None:
        raise ValueError(
            f"No SpaCy token found at char_index={char_index} in '{sentence}'"
        )

    # ── Step 2: Fetch all Lexin candidates (no hardcoding) ───────────────
    candidates = _fetch_candidates(token.text, token.lemma_)
    if not candidates:
        return None

    # ── Step 3: Hard POS filter ───────────────────────────────────────────
    candidates = _filter_by_pos(candidates, token.pos_)
    if not candidates:
        return None

    # ── Step 4: Hard morphological form-existence filter ─────────────────
    candidates = _filter_by_form(candidates, token.text)
    if not candidates:
        return None

    # ── Step 5: Definiteness filter (nouns only) ─────────────────────────
    if token.pos_ in ("NOUN", "PROPN"):
        expected_def = _get_expected_definiteness(token)
        if expected_def is not None:
            candidates = _filter_by_definiteness(
                candidates, token.text, expected_def
            )

    # Single candidate — no scoring needed.
    if len(candidates) == 1:
        c = candidates[0]
        return {"definition": c["definition"], "id": c["id"], "score": 1.0}

    # ── Step 6: Lesk bag-of-words overlap ────────────────────────────────
    lesk = [_lesk_score(c, doc, token) for c in candidates]
    max_lesk = max(lesk)
    if max_lesk > 0:
        top = [c for c, s in zip(candidates, lesk) if s == max_lesk]
        if len(top) == 1:
            c = top[0]
            return {
                "definition": c["definition"],
                "id":         c["id"],
                "score":      float(max_lesk),
            }
        candidates = top  # Narrow to tied leaders before embedding step.

    # ── Step 7: Embedding fallback — raises on any Ollama failure ─────────
    scores = _embedding_scores(candidates, sentence, token, doc)
    best   = int(np.argmax(scores))
    c      = candidates[best]
    return {
        "definition": c["definition"],
        "id":         c["id"],
        "score":      float(scores[best]),
    }