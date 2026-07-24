#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_pivots.py
================
Runs sibling-candidate generation across ALL pivots in multisense_words.json
and attaches cheap per-sense quality flags, so pivots can be triaged
most-promising-first instead of checked one at a time.

Flags (heuristic, for TRIAGE not ground truth — thresholds below are a
first guess, calibrate against words you already know like affär/back):

  - SYNONYM_PILEUP: top-3 candidates are near-duplicates of EACH OTHER
    (mutual similarity too high) — suggests no real category, just the
    same word restated (this is what killed 'back' sense 1 / bakåt).
  - SCATTERED: top-10 candidates don't cohere with each other despite each
    being similar to the pivot — suggests the definition spans >1 domain
    (this is what happened to 'back' sense 4 / car-or-boat gear).
  - CLOSED_CLASS_POS: pivot sense is adverb/preposition/etc — these tend
    to have small synonym sets rather than open thematic fields, so they
    may just be bad pivot material regardless of tuning.
  - NOT_EMBEDDED / TOO_FEW_CANDIDATES: data gaps, not quality judgments.

PATCHED (Wiktionary on-demand embedding): multisense_words.json can now
contain senses tagged "source": "wiktionary" (added by
generate_multisense_json.py as a fallback for words Lexin alone didn't
have 4 senses for) that were never part of full_lexicon.json /
embed_lexicon.py's run, so they start out missing from the base
embeddings matrix -- which used to mean they were silently skipped
(NOT_EMBEDDED) and never scored or offered to the LLM at all.
embed_missing_senses() embeds them on demand, using the same
model/endpoint/prefix/text-cleaning as embed_lexicon.py so the vectors
land in a comparable space, and appends them to the in-memory matrix.
They're deliberately never added to full_lexicon.json itself, so
get_candidates()'s existing Lexin-only filter keeps them out of the
CANDIDATE pool for other pivots -- they can be searched FROM (as a
pivot's own sense) but never suggested TO another pivot as a sibling.

Usage:
  python score_pivots.py                       # full report, sorted most-promising first
  python score_pivots.py --word affär          # verbose single-pivot view
  python score_pivots.py --word back           # sanity-check against a known-mixed case
  python score_pivots.py --sort-by problems    # worst-flagged first (default is promise)
  python score_pivots.py --pileup-threshold 0.8 --scatter-threshold 0.2   # override guessed thresholds
"""

import json
import os
import re
import time
import argparse
import numpy as np
import requests

MULTISENSE_FILE = "multisense_words.json"
MATRIX_FILE = "embeddings.npy"
META_FILE = "embeddings_meta.json"
JSONL_FILE = "embeddings.jsonl"
LEXICON_FILE = "full_lexicon.json"

# Wiktionary on-demand embedding -- must match embed_lexicon.py's settings
# (MODEL, OLLAMA_URL, OLLAMA_PREFIX, NUM_GPU) or the new vectors won't land
# in a comparable space to the rest of the matrix. Copy-pasted rather than
# imported, so if embed_lexicon.py's settings ever change, these need
# updating by hand too -- a real footgun, just not one worth a shared-
# constants refactor right now.
WIKTIONARY_EMBED_CACHE = "wiktionary_embeddings.jsonl"
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text-v2-moe"
EMBED_PREFIX = ""  # must match embed_lexicon.py's OLLAMA_PREFIX
EMBED_NUM_GPU = 2  # must match embed_lexicon.py's NUM_GPU
EMBED_BATCH_SIZE = 32

# SALDO/Lexin POS tags for closed-class words — inferred from general
# convention, NOT verified against your actual tag inventory. Worth
# double-checking against the real tags seen in full_lexicon.json (we've
# only directly observed nn, vb, av, ab, pp so far in this conversation).
CLOSED_CLASS_POS = {"ab", "pp", "in", "kn", "sn", "pn", "ie"}


def load_embeddings():
    if os.path.exists(MATRIX_FILE) and os.path.exists(META_FILE):
        matrix = np.load(MATRIX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"Loaded finalized embeddings: {matrix.shape}")
        return matrix, meta
    if not os.path.exists(JSONL_FILE):
        raise FileNotFoundError("No embeddings found. Run embed_lexicon.py first.")
    print("Building matrix from embeddings.jsonl...")
    vectors, meta = [], []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            vectors.append(rec["embedding"])
            meta.append({"id": rec["id"], "text": rec["text"]})
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms
    print(f"Built matrix: {matrix.shape}")
    return matrix, meta


def load_lexicon():
    with open(LEXICON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_for_embedding(definition: str) -> str:
    """Same cleanup as embed_lexicon.py's clean_for_embedding() -- strips
    Lexin's bare-digit cross-reference parens, e.g. 'bra, fin (2)' ->
    'bra, fin'. Duplicated here rather than imported so this module
    doesn't depend on embed_lexicon.py's script-level state; keep the two
    in sync if either changes."""
    cleaned = re.sub(r"\s*\(\d+\)", "", definition)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned if cleaned else definition


def _load_wiktionary_embed_cache() -> dict:
    cache = {}
    if os.path.exists(WIKTIONARY_EMBED_CACHE):
        with open(WIKTIONARY_EMBED_CACHE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache[rec["id"]] = rec
                except Exception:
                    continue  # tolerate a truncated last line from a crash
    return cache


def embed_missing_senses(multisense: dict, matrix: np.ndarray, meta: list) -> tuple[np.ndarray, list]:
    """
    Finds every sense across multisense_words.json tagged
    "source": "wiktionary" that isn't already in the embeddings matrix,
    embeds it on demand (same model/prefix/cleaning as embed_lexicon.py),
    and appends it to the matrix/meta so get_candidates()/sense_spread()
    can look it up like any other sense.

    Restricted to "source": "wiktionary" specifically, not "any sense
    missing from the matrix" -- a stray un-embedded LEXIN sense would
    indicate a real gap in the dedicated embed_lexicon.py run and should
    surface as NOT_EMBEDDED for you to investigate, not get silently
    patched over here.

    Caches results to WIKTIONARY_EMBED_CACHE (same append-only-JSONL
    convention as embeddings.jsonl) so repeat runs don't re-hit Ollama for
    senses already embedded.
    """
    id_to_index = {m["id"]: i for i, m in enumerate(meta)}

    missing = []
    seen_ids = set()
    for word, senses in multisense.items():
        for sense in senses:
            sid = sense.get("id")
            if not sid or sid in id_to_index or sid in seen_ids:
                continue
            if sense.get("source") != "wiktionary":
                continue
            missing.append(sense)
            seen_ids.add(sid)

    if not missing:
        return matrix, meta

    cache = _load_wiktionary_embed_cache()
    still_missing = [s for s in missing if s["id"] not in cache]
    print(f"  {len(missing)} Wiktionary-sourced senses need embeddings "
          f"({len(missing) - len(still_missing)} already cached, {len(still_missing)} to fetch)...")

    if still_missing:
        session = requests.Session()
        with open(WIKTIONARY_EMBED_CACHE, "a", encoding="utf-8") as out:
            for batch_start in range(0, len(still_missing), EMBED_BATCH_SIZE):
                batch = still_missing[batch_start:batch_start + EMBED_BATCH_SIZE]
                texts = [_clean_for_embedding(s["definition"]) for s in batch]
                payload = {
                    "model": EMBED_MODEL,
                    "input": [EMBED_PREFIX + t for t in texts],
                    "options": {"num_gpu": EMBED_NUM_GPU},
                }
                try:
                    r = session.post(EMBED_URL, json=payload, timeout=60)
                    r.raise_for_status()
                    data = r.json()
                    vecs = data.get("embeddings") or [data.get("embedding")]
                    if not vecs or vecs[0] is None:
                        raise KeyError(f"Unexpected Ollama response: {list(data.keys())}")
                except Exception as e:
                    print(f"  [Warning] Failed to embed a batch of {len(batch)} Wiktionary senses: {e}. "
                          f"Retrying once...")
                    time.sleep(1)
                    try:
                        r = session.post(EMBED_URL, json=payload, timeout=60)
                        r.raise_for_status()
                        data = r.json()
                        vecs = data.get("embeddings") or [data.get("embedding")]
                    except Exception as e2:
                        print(f"  [Warning] Batch failed again, skipping {len(batch)} senses "
                              f"(will stay NOT_EMBEDDED): {e2}")
                        continue

                for sense, text, vec in zip(batch, texts, vecs):
                    rec = {"id": sense["id"], "text": text, "embedding": vec}
                    cache[sense["id"]] = rec
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()

    new_vectors, new_meta = [], []
    for s in missing:
        rec = cache.get(s["id"])
        if not rec:
            continue  # embedding failed even after retry -- stays NOT_EMBEDDED downstream
        new_vectors.append(rec["embedding"])
        new_meta.append({"id": rec["id"], "text": rec["text"]})

    if not new_vectors:
        return matrix, meta

    new_matrix = np.array(new_vectors, dtype=np.float32)
    norms = np.linalg.norm(new_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    new_matrix = new_matrix / norms  # same pre-normalization as finalize()

    matrix = np.vstack([matrix, new_matrix])
    meta = meta + new_meta
    print(f"  Added {len(new_vectors)} Wiktionary-sourced vectors to the matrix "
          f"(now {matrix.shape[0]} total).")
    return matrix, meta


def get_candidates(sid, pivot_baseform, matrix, meta, id_to_index, lexicon, top_k):
    idx = id_to_index[sid]
    vec = matrix[idx]
    sims = matrix @ vec
    order = np.argsort(-sims)

    results = []
    for j in order:
        cand_id = meta[j]["id"]
        if cand_id == sid:
            continue
        cand_entry = lexicon.get(cand_id)
        if not cand_entry:
            continue
        if cand_entry["baseform"].lower() == pivot_baseform.lower():
            continue
        results.append({
            "id": cand_id,
            "idx": int(j),
            "baseform": cand_entry["baseform"],
            "pos": cand_entry["part_of_speech"],
            "definition": cand_entry["definition"],
            "score": float(sims[j]),
        })
        if len(results) >= top_k:
            break
    return results


def mutual_sim(indices, matrix):
    vecs = matrix[indices]
    sim = vecs @ vecs.T
    n = len(indices)
    if n < 2:
        return None
    off_diag_sum = sim.sum() - np.trace(sim)
    return float(off_diag_sum / (n * (n - 1)))


def contains_word(haystack: str, needle: str) -> bool:
    """Whole-word (not substring) check — 'bakåt' should match in
    'går ... bakåt' but NOT inside 'bakåtsträvande' (no word boundary)."""
    if not needle:
        return False
    return re.search(rf"\b{re.escape(needle.lower())}\b", haystack.lower()) is not None


def sense_spread(sense_ids, matrix, id_to_index, close_threshold):
    """
    Pairwise similarity among the PIVOT'S OWN senses (not candidates).
    High similarity between two senses of the same pivot predicts their
    candidate pools will overlap (collision risk) and, independently,
    makes for a less surprising puzzle — the whole mechanic depends on
    the senses feeling unrelated to each other.

    close_threshold is a first guess, uncalibrated — same situation as
    the earlier scatter/pileup thresholds. Check real output (e.g. does
    'stämma"s voice vs musical-part senses score higher than voice vs
    sue?) before trusting the default.
    """
    valid = [(sid, id_to_index[sid]) for sid in sense_ids if sid in id_to_index]
    if len(valid) < 2:
        return None, [], []

    idxs = [i for _, i in valid]
    vecs = matrix[idxs]
    sim = vecs @ vecs.T
    n = len(valid)

    off_diag_sum = sim.sum() - np.trace(sim)
    avg_spread = float(off_diag_sum / (n * (n - 1)))

    all_pairs = []
    close_pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            score = float(sim[a, b])
            pair = (valid[a][0], valid[b][0], score)
            all_pairs.append(pair)
            if score > close_threshold:
                close_pairs.append(pair)
    all_pairs.sort(key=lambda p: -p[2])
    close_pairs.sort(key=lambda p: -p[2])

    return avg_spread, close_pairs, all_pairs


def score_sense(candidates, matrix, pivot_pos, pivot_baseform, scatter_threshold, cross_ref_min_hits):
    """
    SCATTERED: top-3 candidates (the ones actually seen first) don't cohere
    with each other, despite each being individually close to the pivot —
    narrowed from top-10 after finding that a coherent tail cluster (e.g.
    boat parts) can mask genuine incoherence at the top of the list.

    SYNONYM_PILEUP: structural, not magnitude-based. Checks whether a top
    candidate's OWN definition literally names the pivot or another top
    candidate by baseform — the dictionary's own way of saying "this is
    just another word for that." Replaces an earlier magnitude-based
    version that couldn't distinguish a synonym pileup (high mutual
    similarity for the wrong reason) from a genuinely tight, healthy
    category (also high mutual similarity) — both produced overlapping
    cosine ranges (~0.63 vs ~0.76) with no clean cutoff between them.
    """
    flags = []
    metrics = {}

    if pivot_pos in CLOSED_CLASS_POS:
        flags.append("CLOSED_CLASS_POS")

    if len(candidates) < 3:
        flags.append("TOO_FEW_CANDIDATES")
        return flags, metrics

    top3 = candidates[:3]

    top3_sim = mutual_sim([c["idx"] for c in top3], matrix)
    metrics["top3_mutual_sim"] = top3_sim
    if top3_sim is not None and top3_sim < scatter_threshold:
        flags.append("SCATTERED")

    cross_ref_hits = 0
    cross_ref_pairs = []
    for c1 in top3:
        if contains_word(c1["definition"], pivot_baseform):
            cross_ref_hits += 1
            cross_ref_pairs.append(f"{c1['baseform']} -> PIVOT({pivot_baseform})")
        for c2 in top3:
            if c1 is c2:
                continue
            if contains_word(c1["definition"], c2["baseform"]):
                cross_ref_hits += 1
                cross_ref_pairs.append(f"{c1['baseform']} -> {c2['baseform']}")

    metrics["cross_ref_hits"] = cross_ref_hits
    metrics["cross_ref_pairs"] = cross_ref_pairs
    if cross_ref_hits >= cross_ref_min_hits:
        flags.append("SYNONYM_PILEUP")

    return flags, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--word", default=None, help="Only score this pivot, verbose output")
    ap.add_argument("--sort-by", choices=["problems", "promise"], default="promise")
    ap.add_argument("--pileup-threshold", type=float, default=0.75, help=argparse.SUPPRESS)  # deprecated, unused
    ap.add_argument("--scatter-threshold", type=float, default=0.3,
                     help="Below this top-3 mutual similarity, flag SCATTERED (default 0.3, "
                          "calibrated so back's known-bad sense (0.240) flags and known-good senses (0.61-0.76) don't)")
    ap.add_argument("--cross-ref-min-hits", type=int, default=2,
                     help="Minimum candidate<->pivot or candidate<->candidate definition "
                          "cross-references before flagging SYNONYM_PILEUP")
    ap.add_argument("--close-sense-threshold", type=float, default=0.5,
                     help="Above this similarity between two of the PIVOT's OWN senses, "
                          "flag them as too close to each other (uncalibrated first guess)")
    args = ap.parse_args()

    with open(MULTISENSE_FILE, "r", encoding="utf-8") as f:
        multisense = json.load(f)

    matrix, meta = load_embeddings()
    lexicon = load_lexicon()
    matrix, meta = embed_missing_senses(multisense, matrix, meta)
    id_to_index = {m["id"]: i for i, m in enumerate(meta)}

    words_to_check = [args.word] if args.word else list(multisense.keys())
    pivot_reports = []

    for word in words_to_check:
        if word not in multisense:
            print(f"'{word}' not in {MULTISENSE_FILE}, skipping.")
            continue

        sense_reports = []
        for sense in multisense[word]:
            sid = sense["id"]
            if sid not in id_to_index:
                sense_reports.append({
                    "id": sid, "definition": sense["definition"],
                    "flags": ["NOT_EMBEDDED"], "metrics": {}, "candidates": [],
                })
                continue

            pivot_entry = lexicon.get(sid, {})
            pivot_pos = pivot_entry.get("part_of_speech") or sense.get("part_of_speech")
            pivot_baseform = pivot_entry.get("baseform", word)

            candidates = get_candidates(sid, pivot_baseform, matrix, meta, id_to_index, lexicon, args.top_k)
            flags, metrics = score_sense(candidates, matrix, pivot_pos, pivot_baseform,
                                          args.scatter_threshold, args.cross_ref_min_hits)

            sense_reports.append({
                "id": sid, "pos": pivot_pos, "definition": sense["definition"],
                "flags": flags, "metrics": metrics, "candidates": candidates,
            })

        problem_count = sum(len(sr["flags"]) for sr in sense_reports)

        all_sense_ids = [sr["id"] for sr in sense_reports]
        avg_spread, close_pairs, all_pairs = sense_spread(all_sense_ids, matrix, id_to_index, args.close_sense_threshold)
        problem_count += len(close_pairs)

        pivot_reports.append({
            "word": word, "senses": sense_reports, "problem_count": problem_count,
            "avg_spread": avg_spread, "close_pairs": close_pairs, "all_pairs": all_pairs,
        })

    pivot_reports.sort(key=lambda p: p["problem_count"], reverse=(args.sort_by == "problems"))

    if args.word:
        for p in pivot_reports:
            print(f"\n########## {p['word']} (problem_count={p['problem_count']}) ##########")
            print(f"  Own-sense spread: avg_pairwise_sim={p['avg_spread']:.3f}"
                  if p['avg_spread'] is not None else "  Own-sense spread: n/a (fewer than 2 senses embedded)")
            if p["all_pairs"]:
                print(f"  All sense pairs (threshold={args.close_sense_threshold}):")
                for a, b, score in p["all_pairs"]:
                    flag = "  <-- CLOSE" if score > args.close_sense_threshold else ""
                    print(f"    {score:.3f}  {a}  <->  {b}{flag}")
            for sr in p["senses"]:
                print(f"\n=== {sr['id']} [{sr.get('pos')}]: {sr['definition']!r} "
                      f"flags={sr['flags']} metrics={sr.get('metrics')} ===")
                for c in sr["candidates"]:
                    print(f"  {c['score']:.3f}  {c['baseform']:20s} [{c['pos']}]  {c['definition']}")
        return

    print(f"\n{'WORD':20s} {'PROBLEMS':9s} {'SPREAD':7s} FLAGS PER SENSE")
    for p in pivot_reports:
        flag_summary = " | ".join(
            f"{sr['id'].split('..')[-1]}:{','.join(sr['flags']) or 'ok'}" for sr in p["senses"]
        )
        spread_str = f"{p['avg_spread']:.3f}" if p['avg_spread'] is not None else "n/a"
        close_flag = f" CLOSE({len(p['close_pairs'])})" if p["close_pairs"] else ""
        print(f"{p['word']:20s} {p['problem_count']:<9d} {spread_str:7s} {flag_summary}{close_flag}")


if __name__ == "__main__":
    main()