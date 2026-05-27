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
EMBED_MODEL = "nomic-embed-text-v2-moe"

# Multiplier applied to candidates whose Lexin POS matches SpaCy's guess.
# SpaCy is usually right, so we reward agreement — but we never hard-reject
# mismatching candidates, because SpaCy can be wrong on ambiguous tokens
# (e.g. sentence-initial "Får" tagged VERB when it is actually a noun).
# A value of 1.5 means the embedding must differ by >50 % to override SpaCy.
POS_BONUS: float = 1.5

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
    Call the local Ollama embedding API.

    Raises
    ------
    requests.ConnectionError   : Ollama is not running.
    requests.HTTPError         : Non-2xx response from Ollama.
    KeyError                   : Unexpected response format from Ollama.
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

    q_form = (
        f"or(equals|languages.baseform|{wf}"
        f"||inflectionTable(equals|writtenForm|{wf}))"
    )
    queries: set[str] = {q_form}

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

def _preferred_by_pos(candidates: list[dict], spacy_pos: str) -> set[str]:
    """
    Return the set of candidate IDs whose Lexin POS is compatible with
    SpaCy's Universal POS tag.  Candidates with POS '?' are never preferred.

    This is deliberately *advisory* — callers use the returned set to apply
    a scoring bonus, not to hard-filter.  SpaCy can misparse highly ambiguous
    tokens (e.g. "får" at sentence-initial position), so we must never
    eliminate candidates solely on the basis of its POS guess.
    """
    allowed = _POS_MAP.get(spacy_pos, [])
    if not allowed:
        # Unknown SpaCy POS — prefer everything except untagged phrasal entries.
        return {c["id"] for c in candidates if c["pos"] != "?"}
    return {
        c["id"]
        for c in candidates
        if c["pos"] != "?" and any(p in c["pos"] for p in allowed)
    }


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

        # Check "indef"/"obest" BEFORE "def"/"best" because "indef" contains
        # "def" as a substring — checking "def" first would misclassify.
        msd = (infl.get("lexinMsd", "") + " " + infl.get("msd", "")).lower()

        if "indef" in msd or "obest" in msd:
            return "Ind"
        if "best" in msd or ".def" in msd:
            return "Def"

        return "other"

    return None


def _filter_by_form(candidates: list[dict], word_form: str) -> list[dict]:
    """
    Hard form-existence filter: remove candidates that have a non-empty
    inflection table that does NOT contain *word_form* at all — meaning
    the surface word form is morphologically impossible for that sense.

    Candidates without any inflection data pass through unchanged.
    """
    def _ok(c: dict) -> bool:
        if not c.get("inflections"):
            return True
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

    If the filter would remove all candidates the original list is returned
    unchanged — this function never produces an empty set.
    """
    def _compatible(c: dict) -> bool:
        role = _form_role(c, word_form)
        if role is None or role == "other":
            return True
        if role == "baseform":
            return expected_def == "Ind"
        return role == expected_def

    filtered = [c for c in candidates if _compatible(c)]
    return filtered if filtered else candidates


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def _lesk_score(candidate: dict, doc, target_token) -> int:
    """
    Classic Lesk word-overlap score: count how many lemmas of the sentence's
    content words also appear (as raw tokens) in the definition text.
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

_NOMINAL_DEPS = {"nsubj", "nsubj:pass", "obj", "iobj", "obl", "appos"}

def _filter_by_dependency_role(candidates, token):
    """Keep only noun candidates if token is a nominal argument."""
    if token.dep_ in _NOMINAL_DEPS:
        filtered = [c for c in candidates if c["pos"].startswith("nn")]
        return filtered if filtered else candidates
    return candidates

def _embedding_scores(
    candidates: list[dict],
    sentence: str,
    target_token,
    doc,
) -> np.ndarray:
    """
    Cosine-similarity scores between a cloze query and each definition.

    The query blanks out the target word and foregrounds surrounding
    content-word lemmas as a semantic hint, biasing the embedding toward
    the meaning implied by context rather than the surface form.
    """
    blanked = (
        sentence[: target_token.idx]
        + "___"
        + sentence[target_token.idx + len(target_token.text) :]
    )

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
    print("\n **** QUERY TEXT START ******* \n ")
    print(all_texts)
    print("\n **** QUERY TEXT END *********\n ")
    embeds    = _get_embedding(all_texts)

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
    print(f"token.text={token.text!r}, token.pos_={token.pos_!r}, token.dep_={token.dep_!r}")
    # ── Step 2: Fetch all Lexin candidates ───────────────────────────────
    candidates = _fetch_candidates(token.text, token.lemma_)
    if not candidates:
        return None

    # ── Step 3: Compute POS-preferred set (advisory — not a hard filter) ─
    #
    # SpaCy's POS tagger can be wrong on highly ambiguous tokens, so we
    # never discard candidates solely because their POS doesn't match.
    # Instead we record which candidates SpaCy agrees with and apply a
    # scoring bonus to them later.  The embedding can still override SpaCy
    # when the contextual evidence is strong enough.
    pos_preferred: set[str] = _preferred_by_pos(candidates, token.pos_)

    # ── Step 4: Hard morphological form-existence filter ─────────────────
    # This filter IS hard because it is based on objective morphological
    # facts (the word form literally cannot occur in that paradigm), not
    # on a probabilistic tagger.
    candidates = _filter_by_form(candidates, token.text)
    if not candidates:
        return None

    candidates = _filter_by_dependency_role(candidates, token)
    if not candidates:
        return None

    # Keep preferred set in sync after form filtering.
    pos_preferred = {cid for cid in pos_preferred
                     if any(c["id"] == cid for c in candidates)}

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

    # ── Step 6: Lesk bag-of-words overlap with POS bonus ─────────────────
    lesk_raw = [_lesk_score(c, doc, token) for c in candidates]
    lesk = [
        s * (POS_BONUS if c["id"] in pos_preferred else 1.0)
        for c, s in zip(candidates, lesk_raw)
    ]
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
        candidates = top

    # ── Step 7: Embedding fallback with POS bonus ─────────────────────────
    raw_scores = _embedding_scores(candidates, sentence, token, doc)
    scores = np.array([
        s * (POS_BONUS if c["id"] in pos_preferred else 1.0)
        for c, s in zip(candidates, raw_scores)
    ])
    best = int(np.argmax(scores))
    c    = candidates[best]
    return {
        "definition": c["definition"],
        "id":         c["id"],
        "score":      float(scores[best]),
    }