#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_full_lexicon.py
======================
Fetches the ENTIRE Lexin lexicon (not just multisense pivots), for use as
the embedding corpus for sibling-finding. Partitions by starting letter to
stay under Elasticsearch's 10,000 from+size window (confirmed necessary:
naive from/size pagination 500s past from=10000).

Query shape confirmed by testing:
  languages(and(equals|lang|swe||startswith|baseform|<letter>))
- Must use sub-query syntax `languages(...)` since `languages` is a
  collection field (dot-path like `languages.baseform` silently matches
  across ANY language object in the collection, not just Swedish).
- `startswith` appears to match on tokenized words within baseform, not
  the whole string — multi-word entries ("Folkpartiet Liberalerna") get
  matched under every letter one of their words starts with. This causes
  bucket overlap (~35k raw hits vs ~29k true corpus size), which is fine:
  we dedupe by sense id, and filter out multi-word/compound entries anyway
  since they're not usable puzzle words.
"""

import json
import os
import time
import unicodedata
import requests

KARP_API = "https://spraakbanken4.it.gu.se/karp/v7/query/lexin"
OUTPUT_FILE = "full_lexicon.json"

ALPHABET = "abcdefghijklmnopqrstuvwxyzåäö"

# Swedish alphabet + hyphen (for compounds like "A-aktie") + é/É, which is
# common enough in everyday Swedish loanwords (idé, kafé, kliché) to keep.
# Anything else (à, è, ü, ñ, ç, ô, digits, apostrophes, etc.) gets rejected —
# whitelist rather than blacklist, so we don't have to keep discovering new
# stray diacritics one at a time.
ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzåäöéABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖÉ-")

def is_usable_baseform(bf: str) -> bool:
    """Reject multi-word phrases, compounds, single letters, and words
    containing non-whitelisted characters (foreign diacritics, invisible
    unicode, digits, etc.)."""
    if " " in bf or "_" in bf:
        return False
    if len(bf) <= 1:
        return False
    if not all(ch in ALLOWED_CHARS for ch in bf):
        return False
    return True

def clean_baseform(bf: str) -> str:
    """Normalize to NFC so combining-character sequences (e.g. bare 'a' +
    combining grave accent) collapse into their precomposed form (e.g. 'à')
    before we test/reject them — otherwise a decomposed diacritic might not
    match the char it visually represents."""
    return unicodedata.normalize("NFC", bf).strip()

def parse_hit(hit: dict) -> dict | None:
    entry = hit.get("entry", {})
    sense = entry.get("sense", {})

    languages = entry.get("languages", [])
    swe = next((l for l in languages if l.get("lang") == "swe"), None)
    if not swe:
        return None

    baseform = swe.get("baseform")
    if isinstance(baseform, list):
        baseform = baseform[0] if baseform else None
    if not baseform or not isinstance(baseform, str):
        return None
    baseform = clean_baseform(baseform)
    if not is_usable_baseform(baseform):
        return None

    sense_id = sense.get("senseid")
    if not sense_id:
        return None

    definition = sense.get("definition", {}).get("text", "").strip()
    if not definition:
        return None

    return {
        "id": sense_id,
        "baseform": baseform,
        "part_of_speech": swe.get("partOfSpeech", "?"),
        "phonetic": swe.get("phoneticForm"),
        "definition": definition,
    }

def fetch_letter_bucket(session: requests.Session, letter: str) -> list[dict]:
    q = f"languages(and(equals|lang|swe||startswith|baseform|{letter}))"

    # First call to get the true total for this bucket
    r = session.get(KARP_API, params={"q": q, "from": 0, "size": 1}, timeout=15)
    r.raise_for_status()
    total = r.json().get("total", 0)

    if total == 0:
        return []
    if total > 9000:
        # Safety margin under the 10k window — shouldn't happen based on
        # our test sweep (max bucket was ~5200), but guard anyway.
        print(f"\n  [Warning] bucket '{letter}' has {total} hits, near the ES window limit. "
              f"Consider sub-partitioning by two-letter prefix.")

    r = session.get(KARP_API, params={"q": q, "from": 0, "size": min(total, 9000)}, timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", [])

    parsed = []
    for hit in hits:
        p = parse_hit(hit)
        if p:
            parsed.append(p)
    return parsed

def main():
    session = requests.Session()
    result = {}  # keyed by sense id, for dedup across overlapping buckets
    start_time = time.time()

    for letter in ALPHABET:
        print(f"Fetching bucket '{letter}'...", end="", flush=True)
        try:
            senses = fetch_letter_bucket(session, letter)
            new_count = 0
            for s in senses:
                if s["id"] not in result:
                    result[s["id"]] = s
                    new_count += 1
            print(f" {len(senses)} hits, {new_count} new (running total: {len(result)})")
        except Exception as e:
            print(f" [Error] {e}")
        time.sleep(0.1)

    print(f"\nWriting {len(result)} unique senses to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    duration = time.time() - start_time
    print(f"Done in {duration:.1f}s.")

if __name__ == "__main__":
    main()