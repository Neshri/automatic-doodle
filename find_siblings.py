#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_siblings.py
=================
For a given multisense pivot word, ranks nearest-neighbor sibling candidates
per sense using precomputed embeddings — for eyeballing and calibrating a
"lagom" similarity band before we hardcode any thresholds.

Usage:
  python find_siblings.py --word affär
  python find_siblings.py --word affär --top-k 30
  python find_siblings.py --word affär --min-sim 0.4 --max-sim 0.9 --pos-strict

No filtering thresholds are applied by default — this is meant to be run
first with everything visible, so you can see where "genuinely good
sibling" turns into "basically a synonym" (too high) or "barely related"
(too low) along the ranking, THEN set --min-sim/--max-sim from what you see.
"""

import json
import os
import argparse
import numpy as np
import requests

MULTISENSE_FILE = "multisense_words.json"
MATRIX_FILE = "embeddings.npy"
META_FILE = "embeddings_meta.json"
JSONL_FILE = "embeddings.jsonl"
LEXICON_FILE = "full_lexicon.json"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text-v2-moe"
NUM_GPU = 0


def get_live_embedding(text: str) -> np.ndarray:
    """Embed a single text fresh via Ollama, e.g. for testing a query
    variant (like definition-only) without re-embedding the whole corpus."""
    payload = {"model": MODEL, "input": [text], "options": {"num_gpu": NUM_GPU}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    vec = np.array(data["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def load_embeddings():
    """Prefer finalized npy+meta; fall back to building from JSONL directly.
    The JSONL fallback is intentional — it lets you test against a still-
    in-progress embedding run without waiting for --finalize."""
    if os.path.exists(MATRIX_FILE) and os.path.exists(META_FILE):
        matrix = np.load(MATRIX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"Loaded finalized embeddings: {matrix.shape}")
        return matrix, meta

    if not os.path.exists(JSONL_FILE):
        raise FileNotFoundError("No embeddings found. Run embed_lexicon.py first.")

    print("No finalized embeddings.npy found — building from embeddings.jsonl directly.")
    vectors, meta = [], []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue  # tolerate a truncated last line from an in-progress run
            vectors.append(rec["embedding"])
            meta.append({"id": rec["id"], "text": rec["text"]})

    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms
    print(f"Built in-memory matrix from JSONL: {matrix.shape}")
    return matrix, meta


def load_lexicon():
    """embeddings_meta.json only has id + embed text — POS and baseform for
    filtering/display come from full_lexicon.json."""
    if not os.path.exists(LEXICON_FILE):
        raise FileNotFoundError(f"{LEXICON_FILE} not found — needed for POS/baseform lookup.")
    with open(LEXICON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="slag", help="Pivot word from multisense_words.json")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-sim", type=float, default=None,
                     help="Filter out candidates below this cosine similarity (too unrelated)")
    ap.add_argument("--max-sim", type=float, default=None,
                     help="Filter out candidates above this cosine similarity (too obviously synonymous)")
    ap.add_argument("--pos-strict", action="store_true",
                     help="Only show candidates matching the pivot sense's part of speech")
    ap.add_argument("--definition-only-query", action="store_true",
                     help="Re-embed the pivot sense live using ONLY its definition text "
                          "(no surface/baseform prefix), to test whether the shared "
                          "headword token is polluting similarity scores")
    args = ap.parse_args()

    if not os.path.exists(MULTISENSE_FILE):
        raise FileNotFoundError(f"{MULTISENSE_FILE} not found.")
    with open(MULTISENSE_FILE, "r", encoding="utf-8") as f:
        multisense = json.load(f)

    if args.word not in multisense:
        print(f"'{args.word}' not found in {MULTISENSE_FILE}.")
        return

    pivot_senses = multisense[args.word]
    matrix, meta = load_embeddings()
    lexicon = load_lexicon()
    id_to_index = {m["id"]: i for i, m in enumerate(meta)}

    top_candidates_per_sense = {}  # for the cross-sense collision check below

    for sense in pivot_senses:
        sid = sense["id"]

        if args.definition_only_query:
            # Live re-embed using ONLY the definition — no surface/baseform
            # prefix — so the pivot's own headword token can't leak into
            # the query vector and pull in candidates that merely mention
            # "affär" in their own gloss (e.g. "butik: affär (2)").
            vec = get_live_embedding(sense["definition"])
            pivot_pos = lexicon.get(sid, {}).get("part_of_speech")
            pivot_baseform = lexicon.get(sid, {}).get("baseform", args.word)
        else:
            if sid not in id_to_index:
                print(f"\n=== {sid}: NOT FOUND in embeddings "
                      f"(not embedded yet, or filtered out of full_lexicon.json) ===")
                continue
            idx = id_to_index[sid]
            vec = matrix[idx]
            pivot_entry = lexicon.get(sid, {})
            pivot_pos = pivot_entry.get("part_of_speech")
            pivot_baseform = pivot_entry.get("baseform", args.word)

        sims = matrix @ vec  # cosine similarity — rows are pre-normalized

        order = np.argsort(-sims)  # descending

        results = []
        for j in order:
            cand_id = meta[j]["id"]
            if cand_id == sid:
                continue
            cand_entry = lexicon.get(cand_id)
            if not cand_entry:
                continue  # candidate got embedded but later filtered out of full_lexicon somehow
            if cand_entry["baseform"].lower() == pivot_baseform.lower():
                continue  # this is the pivot word's OWN other sense, not a sibling

            score = float(sims[j])
            if args.min_sim is not None and score < args.min_sim:
                continue
            if args.max_sim is not None and score > args.max_sim:
                continue

            pos_match = (cand_entry["part_of_speech"] == pivot_pos)
            if args.pos_strict and not pos_match:
                continue

            results.append({
                "id": cand_id,
                "baseform": cand_entry["baseform"],
                "pos": cand_entry["part_of_speech"],
                "pos_match": pos_match,
                "definition": cand_entry["definition"],
                "score": score,
            })
            if len(results) >= args.top_k:
                break

        top_candidates_per_sense[sid] = results

        print(f"\n=== {sid} [{pivot_pos}]: {sense['definition']!r} ===")
        for r in results:
            flag = "" if r["pos_match"] else "  (POS mismatch)"
            print(f"  {r['score']:.3f}  {r['baseform']:20s} [{r['pos']}]  {r['definition']}{flag}")

    print("\n=== Cross-sense collisions (same candidate scoring in top-K for 2+ senses) ===")
    seen = {}
    for sid, results in top_candidates_per_sense.items():
        for r in results:
            seen.setdefault(r["baseform"], set()).add(sid)
    collisions = {bf: sids for bf, sids in seen.items() if len(sids) > 1}
    if collisions:
        for bf, sids in collisions.items():
            print(f"  '{bf}' appears for: {sorted(sids)}")
    else:
        print("  None found.")


if __name__ == "__main__":
    main()