#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed_lexicon.py
=================
Embeds every sense in full_lexicon.json using Ollama + nomic-embed-text-v2-moe.

Checkpointed via append-only JSONL: safe to interrupt (laptop sleep, crash,
Ctrl+C) and resume — already-embedded ids are skipped on restart.

Once all entries are embedded, run with --finalize to pack the JSONL into
a single numpy matrix + metadata list for fast cosine-similarity search.

Embedding text = definition only, with Lexin's parenthetical sense-number
cross-references (e.g. "(2)" in "bra, fin (2)") stripped out. No surface
form or baseform prefix — this pipeline has no context sentence to
disambiguate against, so a prefix only risked pulling in the pivot's own
headword as a spurious shared token with any corpus entry whose gloss
happens to cross-reference it.

NOTE: nomic-embed models are often trained with task-prefix conventions
(e.g. "search_document: " prepended). If definition_finder.py's Ollama
calls use such a prefix, set OLLAMA_PREFIX below to match — otherwise this
runs with no prefix, which may or may not be optimal. Worth checking
retrieval quality after a first pass either way.
"""

import json
import os
import re
import time
import requests
import numpy as np

INPUT_FILE = "full_lexicon.json"
JSONL_FILE = "embeddings.jsonl"
MATRIX_FILE = "embeddings.npy"
META_FILE = "embeddings_meta.json"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text-v2-moe"
OLLAMA_PREFIX = ""  # e.g. "search_document: " — set to match definition_finder.py if it uses one
BATCH_SIZE = 32
NUM_GPU = 0  # matches definition_finder.py's setting; flip to something >0 if nothing else is holding VRAM right now

def clean_for_embedding(definition: str) -> str:
    """Strip Lexin's own cross-reference shorthand — a trailing/inline
    '(N)' pointing to sense N of some other word (e.g. 'bra, fin (2)',
    'affär (2)') — before embedding. This is pure lexicographic notation,
    not semantic content, but it was acting as a shared literal token that
    spuriously boosted similarity between otherwise-unrelated words (e.g.
    'loft: vind (2)' and 'eran: er (2)' scoring close purely because both
    end in '(2)'). Does NOT touch parentheticals with real content, e.g.
    '(av företag i ekonomisk kris)' — only bare-digit parens are stripped.
    """
    cleaned = re.sub(r"\s*\(\d+\)", "", definition)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned if cleaned else definition  # never embed an empty string


def build_embed_text(entry: dict) -> str:
    # No surface/baseform prefix (see prior note on why); definition text
    # is also cleaned of Lexin's parenthetical sense-number cross-refs.
    return clean_for_embedding(entry["definition"])


def get_embeddings_batch(session: requests.Session, texts: list[str]) -> np.ndarray:
    payload = {
        "model": MODEL,
        "input": [OLLAMA_PREFIX + t for t in texts],
        "options": {"num_gpu": NUM_GPU},
    }
    r = session.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "embeddings" in data:
        return np.array(data["embeddings"])
    if "embedding" in data:
        return np.array([data["embedding"]])
    raise KeyError(f"Unexpected Ollama response: {list(data.keys())}")


def load_already_done() -> set[str]:
    done = set()
    if os.path.exists(JSONL_FILE):
        with open(JSONL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["id"])
                except Exception:
                    continue  # tolerate a truncated last line from a crash
    return done


def run_embedding_pass():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lexicon = json.load(f)

    already_done = load_already_done()
    print(f"Loaded {len(lexicon)} total senses. {len(already_done)} already embedded, resuming.")

    remaining = [entry for sid, entry in lexicon.items() if sid not in already_done]
    print(f"{len(remaining)} remaining to embed.")

    session = requests.Session()
    start_time = time.time()

    with open(JSONL_FILE, "a", encoding="utf-8") as out:
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start:batch_start + BATCH_SIZE]
            texts = [build_embed_text(e) for e in batch]

            try:
                vecs = get_embeddings_batch(session, texts)
            except Exception as e:
                print(f"\n[Error embedding batch at {batch_start}]: {e}. Retrying once...")
                time.sleep(1)
                try:
                    vecs = get_embeddings_batch(session, texts)
                except Exception as e2:
                    print(f"[Batch failed again, skipping {len(batch)} entries]: {e2}")
                    continue

            for entry, text, vec in zip(batch, texts, vecs):
                record = {"id": entry["id"], "text": text, "embedding": vec.tolist()}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()  # ensure the whole batch survives a crash/interrupt immediately

            done_count = batch_start + len(batch)
            if done_count % (BATCH_SIZE * 10) == 0 or done_count == len(remaining):
                elapsed = time.time() - start_time
                rate = done_count / elapsed
                eta = (len(remaining) - done_count) / rate if rate > 0 else float("inf")
                print(f"  [{done_count}/{len(remaining)}] {rate:.1f}/s, ETA {eta/60:.1f} min")

    print(f"\nEmbedding pass complete in {(time.time() - start_time)/60:.1f} min.")


def finalize():
    """Pack the JSONL checkpoint into a single numpy matrix + metadata list."""
    if not os.path.exists(JSONL_FILE):
        print(f"Error: {JSONL_FILE} not found — run the embedding pass first.")
        return

    vectors = []
    meta = []
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
    # Pre-normalize rows so cosine similarity later is just a dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    np.save(MATRIX_FILE, matrix)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Finalized: {matrix.shape[0]} vectors of dim {matrix.shape[1]}")
    print(f"  -> {MATRIX_FILE}")
    print(f"  -> {META_FILE}")


if __name__ == "__main__":
    import sys
    if "--finalize" in sys.argv:
        finalize()
    else:
        run_embedding_pass()