"""
definition_finder.py
====================
Swedish Word Sense Disambiguation using the Lexin dictionary (via the Karp API)
and BGE-M3 embeddings (via Ollama) as a fallback scorer.

Public interface
----------------
    find_definition(sentence, word, char_index) -> dict | None
"""

import spacy
import requests
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

nlp = spacy.load("sv_core_news_lg")

KARP_API    = "https://spraakbanken4.it.gu.se/karp/v7/query/lexin"
OLLAMA_API  = "http://localhost:11434/api/embed"
EMBED_MODEL = "bge-m3"

POS_BONUS: float = 1.4

_POS_MAP: dict[str, list[str]] = {
    "NOUN":  ["nn"],
    "PROPN": ["nn"],
    "VERB":  ["vb", "vbm"],
    "AUX":   ["vb", "vbm"],
    "ADJ":   ["jj", "hjj", "av"],
    "ADV":   ["ab"],
    "PRON":  ["pn"],
    "NUM":   ["rg", "ro"],
    "INTJ":  ["in"],
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"  {msg}")

def _log_candidates(label: str, candidates: list[dict], preferred: set[str] | None = None) -> None:
    stars = lambda c: " ★" if preferred and c["id"] in preferred else ""
    rows  = [f"{c['id']:<35} pos={c['pos']:<6} {c['definition']!r}{stars(c)}"
             for c in candidates]
    print(f"  [{label}: {len(candidates)}]")
    for r in rows:
        print(f"    {r}")

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_embedding(texts: list[str]) -> np.ndarray:
    payload = {"model": EMBED_MODEL, "input": texts, "options": {"num_gpu": 0}}
    r = requests.post(OLLAMA_API, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "embeddings" in data:
        return np.array(data["embeddings"])
    if "embedding" in data:
        return np.array([data["embedding"]])
    raise KeyError(f"Unexpected Ollama response: {list(data.keys())}")


def _fetch_candidates(word_form: str, spacy_lemma: str) -> list[dict]:
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
    results:  list[dict] = []

    for q in queries:
        resp = requests.get(KARP_API, params={"q": q, "size": 25})
        resp.raise_for_status()
        for hit in resp.json().get("hits", []):
            entry = hit.get("entry", {})
            sense = entry.get("sense", {})
            sid   = sense.get("senseid", "")
            defn  = sense.get("definition", {}).get("text", "")
            if not sid or not defn or sid in seen_ids:
                continue
            seen_ids.add(sid)

            swe = next(
                (l for l in entry.get("languages", []) if l.get("lang") == "swe"), {}
            )
            raw_baseform = swe.get("baseform", wf)
            if isinstance(raw_baseform, list):
                raw_baseform = raw_baseform[0] if raw_baseform else wf

            results.append({
                "id":          sid,
                "pos":         swe.get("partOfSpeech", "?"),
                "baseform":    raw_baseform,
                "definition":  defn,
                "inflections": entry.get("inflectionTable", []),
            })

    return results


# ---------------------------------------------------------------------------
# Filtering pipeline
# ---------------------------------------------------------------------------

def _preferred_by_pos(candidates: list[dict], spacy_pos: str) -> set[str]:
    allowed = _POS_MAP.get(spacy_pos, [])
    if not allowed:
        return {c["id"] for c in candidates if c["pos"] != "?"}
    return {
        c["id"]
        for c in candidates
        if c["pos"] != "?" and any(p in c["pos"] for p in allowed)
    }


def _form_role(candidate: dict, word_form: str) -> str | None:
    wf = word_form.lower()
    if candidate["baseform"].lower() == wf:
        return "baseform"
    for infl in candidate.get("inflections", []):
        if infl.get("writtenForm", "").lower() != wf:
            continue
        msd = (infl.get("lexinMsd", "") + " " + infl.get("msd", "")).lower()
        if "indef" in msd or "obest" in msd:
            return "Ind"
        if "best" in msd or ".def" in msd:
            return "Def"
        return "other"
    return None


def _filter_by_form(candidates: list[dict], word_form: str) -> list[dict]:
    kept    = [c for c in candidates if not c.get("inflections") or _form_role(c, word_form) is not None]
    removed = len(candidates) - len(kept)
    if removed:
        _log(f"form-filter: removed {removed} whose paradigm cannot produce {word_form!r}")
    return kept


def _get_expected_definiteness(token) -> str | None:
    for child in token.children:
        if child.dep_ in ("det", "quant", "nummod"):
            m = child.morph.to_dict()
            if "Definite" in m:
                return m["Definite"]
    return token.morph.to_dict().get("Definite")


def _filter_by_definiteness(candidates, word_form, expected_def):
    def _ok(c):
        role = _form_role(c, word_form)
        if role is None or role == "other":
            return True
        return (role == "baseform" and expected_def == "Ind") or role == expected_def

    kept    = [c for c in candidates if _ok(c)]
    removed = len(candidates) - len(kept)
    if removed:
        _log(f"def-filter:  removed {removed} incompatible with Definite={expected_def!r}")
    return kept if kept else candidates


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def _lesk_score(candidate: dict, context_lemmas: set[str]) -> tuple[int, set]:
    def_tokens = set(candidate["definition"].lower().split())
    overlap    = context_lemmas & def_tokens
    return len(overlap), overlap


def _morph_to_swedish(morph_dict: dict, spacy_pos: str) -> list[str]:
    """
    Translate the most diagnostically useful morphological features into
    natural Swedish phrases.  Returns a list of short descriptors that
    can be joined with comma.  Raw UD feature strings are never exposed
    to the embedding model.
    """
    parts = []

    pos_labels = {
        "NOUN": "substantiv", "PROPN": "substantiv",
        "VERB": "verb",       "AUX":   "verb",
        "ADJ":  "adjektiv",   "ADV":   "adverb",
        "PRON": "pronomen",   "NUM":   "räkneord",
        "INTJ": "interjektion",
    }
    if label := pos_labels.get(spacy_pos):
        parts.append(label)

    def_map = {"Ind": "obestämd form", "Def": "bestämd form"}
    if d := def_map.get(morph_dict.get("Definite", "")):
        parts.append(d)

    num_map = {"Sing": "singular", "Plur": "plural"}
    if n := num_map.get(morph_dict.get("Number", "")):
        parts.append(n)

    tense_map = {"Pres": "presens", "Past": "preteritum"}
    if t := tense_map.get(morph_dict.get("Tense", "")):
        parts.append(t)

    vf_map = {"Inf": "infinitiv", "Sup": "supinum", "Part": "particip"}
    if v := vf_map.get(morph_dict.get("VerbForm", "")):
        parts.append(v)

    deg_map = {"Comp": "komparativ", "Sup": "superlativ"}
    if d := deg_map.get(morph_dict.get("Degree", "")):
        parts.append(d)

    return parts


def _embedding_scores(
    candidates:  list[dict],
    sentence:    str,
    target_token,
    doc,
) -> np.ndarray:
    blanked = (
        sentence[: target_token.idx]
        + "___"
        + sentence[target_token.idx + len(target_token.text) :]
    )

    target_lemma = target_token.lemma_.lower()

    # Content-word hint — exclude tokens that share lemma/surface with target
    # to avoid circular context (e.g. "bar … bar … bar" all pointing at bära)
    content_lemmas = list(dict.fromkeys(
        t.lemma_
        for t in doc
        if t.pos_ in ("NOUN", "VERB", "ADJ")
        and t.i != target_token.i
        and not t.is_stop
        and len(t.lemma_) > 2
        and t.lemma_.lower() != target_lemma
        and t.text.lower()   != target_token.text.lower()
    ))

    # Immediate neighbours — keep as-is, including repeated forms of the target
    # word, since their presence is genuine context (e.g. «en ___ bar in»)
    left_words  = " ".join(
        doc[i].text
        for i in range(max(0, target_token.i - 2), target_token.i)
        if not doc[i].is_punct
    )
    right_words = " ".join(
        doc[i].text
        for i in range(target_token.i + 1, min(len(doc), target_token.i + 3))
        if not doc[i].is_punct
    )

    morph_parts = _morph_to_swedish(target_token.morph.to_dict(), target_token.pos_)

    # Build query — only include each clause when it carries real information
    ctx_parts = []
    if left_words or right_words:
        ctx_parts.append(f"omgivning: «{left_words} ___ {right_words}»")
    if content_lemmas:
        ctx_parts.append(f"nyckelord: {', '.join(content_lemmas)}")
    if morph_parts:
        ctx_parts.append(f"ordform: {', '.join(morph_parts)}")

    ctx_str = f" ({'; '.join(ctx_parts)})" if ctx_parts else ""
    query   = f"Texten «{blanked}»{ctx_str}. Det obekanta ordet ___ syftar på:"

    _log(f"embedding query: {query}")

    all_texts = [query] + [c["definition"] for c in candidates]

    embeds    = _get_embedding(all_texts)
    q_vec     = embeds[0]
    d_vecs    = embeds[1:]
    norms     = np.linalg.norm(d_vecs, axis=1) * np.linalg.norm(q_vec)
    norms     = np.where(norms == 0, 1e-10, norms)
    return np.dot(d_vecs, q_vec) / norms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_definition(string: str, word: str, char_index: int) -> dict | None:
    print(f"\n{'─'*60}")
    print(f"  {word!r} @{char_index}  ·  {string!r}")
    print(f"{'─'*60}")

    # Step 1: SpaCy
    doc   = nlp(string)
    token = next(
        (t for t in doc if t.idx <= char_index < t.idx + len(t.text)), None
    )
    if token is None:
        raise ValueError(f"No token at char_index={char_index}")

    morph_str = ", ".join(f"{k}={v}" for k, v in token.morph.to_dict().items()) or "—"
    _log(f"token={token.text!r}  lemma={token.lemma_!r}  pos={token.pos_!r}  "
         f"dep={token.dep_!r}  morph=[{morph_str}]")

    # Step 2: Fetch
    candidates = _fetch_candidates(token.text, token.lemma_)
    _log(f"karp: {len(candidates)} raw hits")

    candidates = [
        c for c in candidates
        if "_" not in c["baseform"] and " " not in c["baseform"]
        or c["baseform"].replace("_", " ").lower() in string.lower()
    ]
    if not candidates:
        _log("→ no candidates after multi-word filter")
        return None

    # Step 3: POS preference
    pos_preferred = _preferred_by_pos(candidates, token.pos_)
    _log(f"pos-preferred: {len(pos_preferred)}/{len(candidates)} match spaCy {token.pos_!r} "
         f"→ {_POS_MAP.get(token.pos_, [])}")

    # Step 4: Form filter
    candidates = _filter_by_form(candidates, token.text)
    if not candidates:
        _log("→ no candidates after form filter")
        return None
    pos_preferred = {cid for cid in pos_preferred if any(c["id"] == cid for c in candidates)}

    # Step 5: Definiteness
    if token.pos_ in ("NOUN", "PROPN"):
        expected_def = _get_expected_definiteness(token)
        _log(f"definiteness expected: {expected_def!r}")
        if expected_def:
            candidates = _filter_by_definiteness(candidates, token.text, expected_def)
            pos_preferred = {cid for cid in pos_preferred if any(c["id"] == cid for c in candidates)}

    _log_candidates("after filters", candidates, preferred=pos_preferred)

    if len(candidates) == 1:
        c = candidates[0]
        _log(f"→ single candidate, returning directly")
        return {"definition": c["definition"], "id": c["id"], "score": 1.0}

    # Step 6: Lesk
    context_lemmas = {
        t.lemma_.lower()
        for t in doc
        if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV")
        and t.i != token.i
        and not t.is_stop
        and len(t.lemma_) > 2
    }
    lesk_pairs   = [_lesk_score(c, context_lemmas) for c in candidates]
    lesk_raw     = [s for s, _ in lesk_pairs]
    lesk_boosted = [
        s * (POS_BONUS if c["id"] in pos_preferred else 1.0)
        for c, s in zip(candidates, lesk_raw)
    ]
    max_lesk = max(lesk_boosted)

    if max_lesk > 0:
        _log(f"lesk context lemmas: {sorted(context_lemmas)}")
        for c, (raw, overlap), boosted in zip(candidates, lesk_pairs, lesk_boosted):
            _log(f"  lesk {boosted:.1f}  overlap={sorted(overlap)}  {c['id']}  {c['definition']!r}")
        top = [c for c, s in zip(candidates, lesk_boosted) if s == max_lesk]
        if len(top) == 1:
            c = top[0]
            _log(f"→ lesk winner: {c['id']}  score={max_lesk}")
            return {"definition": c["definition"], "id": c["id"], "score": float(max_lesk)}
        _log(f"lesk tie at {max_lesk} — falling through to embedding")
        candidates = top
    else:
        _log(f"lesk: all zero (context lemmas: {sorted(context_lemmas)}) — skipping to embedding")

    # Step 7: Embedding
    raw_scores = _embedding_scores(candidates, string, token, doc)
    scores = np.array([
        s * (POS_BONUS if c["id"] in pos_preferred else 1.0)
        for c, s in zip(candidates, raw_scores)
    ])
    best = int(np.argmax(scores))
    for i, (c, s) in enumerate(zip(candidates, scores)):
        mark = " ← WINNER" if i == best else ""
        preferred_mark = " ★" if c["id"] in pos_preferred else ""
        _log(f"  emb {s:+.4f}  {c['id']}{preferred_mark}  {c['definition']!r}{mark}")

    c = candidates[best]
    _log(f"→ embedding winner: {c['id']}  score={scores[best]:.4f}")
    return {"definition": c["definition"], "id": c["id"], "score": float(scores[best])}