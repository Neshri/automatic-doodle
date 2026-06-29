#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_multisense_words.py
=========================
Queries the Karp API for Lexin entries, groups them by Swedish baseform,
and extracts words that have 4 or more senses. The resulting list is sorted
alphabetically and written to `multisense_words.txt` with linebreaks.
"""

import sys
import time
import requests
from collections import defaultdict

KARP_API = "https://spraakbanken4.it.gu.se/karp/v7/query/lexin"
OUTPUT_FILE = "multisense_words.txt"
PAGE_SIZE = 1000

# Alphabet to partition our queries to stay under the 10,000 max_result_window limit.
# We include lowercase, uppercase, and hyphen to capture all Swedish baseforms.
ALPHABET = "abcdefghijklmnopqrstuvwxyzåäö-"

def fetch_all_entries() -> dict[str, dict]:
    """
    Fetches all entries from Karp API by partitioning queries alphabetically.
    Returns a dict mapping entry ID to the entry document to ensure deduplication.
    """
    all_hits = {}
    total_queries = 0

    print("Starting data retrieval from Karp API...")
    start_time = time.time()

    for char in ALPHABET:
        print(f"Fetching entries starting with '{char}'...", end="", flush=True)
        char_hits = 0
        from_offset = 0

        while True:
            # Query for Swedish baseforms starting with the character
            query = f"languages(and(equals|lang|swe||startswith|baseform|{char}))"
            params = {
                "q": query,
                "size": PAGE_SIZE,
                "from": from_offset
            }
            
            try:
                r = requests.get(KARP_API, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"\nError fetching '{char}' at offset {from_offset}: {e}")
                # Simple retry once after a brief sleep
                time.sleep(2)
                r = requests.get(KARP_API, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()

            hits = data.get("hits", [])
            if not hits:
                break

            for hit in hits:
                hit_id = hit.get("id")
                if hit_id:
                    all_hits[hit_id] = hit
            
            char_hits += len(hits)
            total_queries += 1

            if len(hits) < PAGE_SIZE:
                # No more pages for this prefix
                break
            
            from_offset += PAGE_SIZE

        print(f" done (found {char_hits} hits, total unique so far: {len(all_hits)})")

    duration = time.time() - start_time
    print(f"Finished fetching. Made {total_queries} API calls in {duration:.1f} seconds.")
    print(f"Total unique entries retrieved: {len(all_hits)}")
    return all_hits

def group_and_filter_senses(entries: dict[str, dict]) -> list[str]:
    """
    Groups entries by baseform and filters for baseforms with 4 or more senses.
    """
    # Map baseform -> set of sense IDs to ensure unique senses per word
    baseform_senses = defaultdict(set)

    for hit_id, hit in entries.items():
        entry = hit.get("entry", {})
        
        # Get Swedish baseform
        languages = entry.get("languages", [])
        swe = next((l for l in languages if l.get("lang") == "swe"), None)
        if not swe:
            continue

        baseform = swe.get("baseform")
        if not baseform:
            continue
        
        # Handle list format for baseform
        if isinstance(baseform, list):
            baseform = baseform[0] if baseform else None
        
        if not baseform or not isinstance(baseform, str):
            continue

        baseform = baseform.strip()
        if not baseform:
            continue

        # Get sense ID
        sense_id = entry.get("sense", {}).get("senseid")
        if not sense_id:
            continue

        # Use normalized lowercase baseform for grouping, but keep original case for display if needed.
        # Since we want to sort alphabetically, we will store the baseform exactly.
        baseform_senses[baseform].add(sense_id)

    # Filter baseforms with 4 or more unique senses, excluding one-letter words and words with spaces
    multisense_words = []
    for word, senses in baseform_senses.items():
        if len(senses) >= 4 and len(word) > 1 and " " not in word:
            multisense_words.append(word)

    return multisense_words

def main():
    try:
        entries = fetch_all_entries()
        multisense_words = group_and_filter_senses(entries)
        
        # Sort alphabetically
        # Swedish alphabetical sorting generally works with standard sort, 
        # but to be clean we sort using Python's default sort (which handles ÅÄÖ correctly at the end of ASCII/Latin-1 if needed, 
        # or standard locale-independent sort).
        multisense_words.sort()

        print(f"Found {len(multisense_words)} words with 4 or more senses.")
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for word in multisense_words:
                f.write(word + "\n")
                
        print(f"Successfully saved words to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
