#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_lopa.py
============
Standalone diagnostic: why does 'löpa' come back not_found in SALDO?

Checks:
  1. What sense IDs does multisense_words.json actually have for 'löpa'?
  2. Does SALDO's senseID field recognize 'löper..1' (Lexin's form),
     'löpa..1' (infinitive/lemma form), or neither?

Run this from the same folder as multisense_words.json.
"""

import json
import os
import requests

KARP_V7_SALDO = "https://spraakbanken4.it.gu.se/karp/v7/query/saldo"
INPUT_FILE = "multisense_words.json"

WORD = "löpa"

def main():
    session = requests.Session()

    # --- Step 1: inspect Lexin's own sense records for 'löpa' ---
    print(f"=== Step 1: Lexin sense records for '{WORD}' ===")
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            lexin_data = json.load(f)

        if WORD in lexin_data:
            for sense in lexin_data[WORD]:
                print(f"  id={sense.get('id')!r}  pos={sense.get('part_of_speech')!r}  def={sense.get('definition')!r}")
        else:
            print(f"  '{WORD}' not found as a key in {INPUT_FILE} (check spelling/casing).")
    else:
        print(f"  {INPUT_FILE} not found in current directory — skipping this step.")

    # --- Step 2: try both citation forms directly against SALDO ---
    print(f"\n=== Step 2: Direct SALDO senseID lookups ===")
    candidates = [
        "löper..1", "löper..2", "löper..3", "löper..4",  # Lexin's apparent form
        "löpa..1", "löpa..2", "löpa..3", "löpa..4",       # infinitive/lemma form
    ]

    for candidate in candidates:
        try:
            r = session.get(KARP_V7_SALDO, params={"q": f"equals|senseID|{candidate}", "size": 1}, timeout=15)
            r.raise_for_status()
            hits = r.json().get("hits", [])
            status = f"{len(hits)} hit(s)"
            if hits:
                entry = hits[0].get("entry", {})
                status += f"  -> baseform={entry.get('baseform')!r}  primary={entry.get('primary')!r}"
        except Exception as e:
            status = f"ERROR: {e}"
        print(f"  {candidate:15s} -> {status}")

    # --- Step 3: broader search — what senseIDs actually exist for this baseform? ---
    print(f"\n=== Step 3: What SALDO senseIDs exist with baseform variants of '{WORD}'? ===")
    for bf in ["löpa", "löper"]:
        try:
            r = session.get(KARP_V7_SALDO, params={"q": f"equals|baseform|{bf}", "size": 50}, timeout=15)
            r.raise_for_status()
            hits = r.json().get("hits", [])
            print(f"  baseform={bf!r}: {len(hits)} hit(s)")
            for hit in hits:
                entry = hit.get("entry", {})
                print(f"    senseID={entry.get('senseID')!r}  baseform={entry.get('baseform')!r}")
        except Exception as e:
            print(f"  baseform={bf!r}: ERROR: {e}")

if __name__ == "__main__":
    main()