#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_multisense_json.py
===========================
Reads words from `multisense_words.txt`, queries the Karp API for Lexin entries,
extracts and formats their Swedish senses, and writes them to a JSON dictionary
where the word is the key and the list of its senses is the value.

PATCHED: senses are now grouped by lexinID, Lexin's own identifier for a
distinct dictionary entry — NOT by baseform string matching. Verified
directly against raw API output that baseform can be identical for two
unrelated entries (noun "stämma" and verb "stämma" both have
baseform="stämma" but different lexinID: 1144651 vs 1144673), which would
otherwise silently merge them into one pivot.

If a queried word maps to more than one lexinID, each becomes its own
pivot entry, keyed by that entry's rawForm (its actual citation spelling —
e.g. "stämmer" for the verb vs "stämma" for the noun, also verified
directly rather than assumed) rather than the queried word string, since
the queried word may not match every split entry's true citation form.
Words mapping to a single lexinID keep their original key, unchanged.
Each resulting group still needs >=4 senses on its own to be kept.
"""

import json
import os
import time
from collections import defaultdict
import requests

KARP_API = "https://spraakbanken4.it.gu.se/karp/v7/query/lexin"
INPUT_FILE = "multisense_words.txt"
OUTPUT_FILE = "multisense_words.json"

def fetch_senses(session: requests.Session, word: str) -> list[dict]:
    """
    Queries the Karp API for the given word and parses the Swedish senses.
    Groups by lexinID, not baseform string — baseform can be identical for
    two genuinely different dictionary entries (e.g. noun "stämma" and verb
    "stämma" share baseform="stämma" but have different lexinID: 1144651
    vs 1144673). lexinID is Lexin's own authoritative "this is one specific
    headword entry" identifier, and rawForm is the actual citation spelling
    for that entry (verbs are cited by present tense in Lexin, e.g.
    "stämmer" for the verb vs "stämma" for the unrelated noun) — both
    verified directly against raw API output, not inferred.
    """
    params = {
        "q": f"languages(and(equals|lang|swe||equals|baseform|{word}))",
        "size": 50
    }
    
    try:
        r = session.get(KARP_API, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"\n[Warning] Error fetching '{word}': {e}. Retrying once...")
        time.sleep(2)
        r = session.get(KARP_API, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

    senses = []
    hits = data.get("hits", [])
    for hit in hits:
        entry = hit.get("entry", {})
        sense = entry.get("sense", {})
        
        languages = entry.get("languages", [])
        swe = next((l for l in languages if l.get("lang") == "swe"), None)
        if not swe:
            continue
        
        baseform = swe.get("baseform")
        if isinstance(baseform, list):
            baseform = baseform[0] if baseform else None
        
        if not baseform or not isinstance(baseform, str):
            continue
            
        if baseform.strip().lower() != word.strip().lower():
            continue
            
        sense_id = sense.get("senseid")
        if not sense_id:
            continue
            
        definition = sense.get("definition", {}).get("text", "").strip()
        if not definition:
            continue
            
        part_of_speech = swe.get("partOfSpeech", "?")
        phonetic = swe.get("phoneticForm")
        lexin_id = swe.get("lexinID")       # groups senses into distinct dictionary entries
        raw_form = swe.get("rawForm", baseform)  # actual citation spelling for this entry
        
        examples = []
        for ex in sense.get("examples", []):
            if ex.get("lang") == "swe" and ex.get("text"):
                examples.append(ex["text"])
                
        usage = sense.get("usg", [])
        
        senses.append({
            "id": sense_id,
            "lexin_id": lexin_id,
            "raw_form": raw_form,
            "part_of_speech": part_of_speech,
            "definition": definition,
            "phonetic": phonetic,
            "usage": usage,
            "examples": examples
        })
    
    return senses

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(words)} words from {INPUT_FILE}.")
    result = {}
    split_count = 0
    dropped_after_split = 0
    
    session = requests.Session()
    start_time = time.time()
    
    for i, word in enumerate(words, 1):
        print(f"\r[{i}/{len(words)}] Fetching senses for '{word}'...", end="", flush=True)
        try:
            senses = fetch_senses(session, word)
            
            # Deduplicate by sense ID
            unique_senses = []
            seen_ids = set()
            for s in senses:
                if s["id"] not in seen_ids:
                    seen_ids.add(s["id"])
                    unique_senses.append(s)

            # Group by lexinID — Lexin's own ground-truth "distinct dictionary
            # entry" identifier. Two senses can share an identical baseform
            # string while belonging to unrelated entries (verified directly:
            # noun "stämma" and verb "stämma" both have baseform="stämma" but
            # lexinID 1144651 vs 1144673) — grouping by baseform alone would
            # silently merge them.
            by_lexin_id = defaultdict(list)
            for s in unique_senses:
                by_lexin_id[s["lexin_id"]].append(s)

            multi_entry = len(by_lexin_id) > 1
            if multi_entry:
                split_count += 1

            for lexin_id, entry_senses in by_lexin_id.items():
                # Always key by raw_form — Lexin's own citation spelling for
                # this entry — rather than falling back to the queried word
                # for single-entry cases. Keeps the convention consistent
                # regardless of whether a word happened to split or not.
                raw_form = entry_senses[0]["raw_form"]
                key = raw_form
                if len(entry_senses) >= 4:
                    result[key] = entry_senses
                else:
                    if multi_entry:
                        dropped_after_split += 1
                    print(f"\n[Warning] '{key}' has only {len(entry_senses)} valid senses "
                          f"{'(one of multiple distinct entries under this spelling) ' if multi_entry else ''}, skipping...")
        except Exception as e:
            print(f"\n[Error] Failed to process word '{word}': {e}")
        
        # Polite delay between requests
        time.sleep(0.05)
        
    print(f"\nFinished fetching. Writing data to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    duration = time.time() - start_time
    print(f"Successfully compiled {len(result)} pivots to {OUTPUT_FILE} in {duration:.1f} seconds.")
    print(f"  {split_count} queried words mapped to >1 distinct lexinID and were split.")
    print(f"  {dropped_after_split} split-off entries fell below the 4-sense minimum on their own.")

if __name__ == "__main__":
    main()