#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_multisense_json.py
===========================
Reads words from `multisense_words.txt`, queries the Karp API for Lexin entries,
extracts and formats their Swedish senses, and writes them to a JSON dictionary
where the word is the key and the list of its senses is the value.
"""

import json
import os
import time
import requests

KARP_API = "https://spraakbanken4.it.gu.se/karp/v7/query/lexin"
INPUT_FILE = "multisense_words.txt"
OUTPUT_FILE = "multisense_words.json"

def fetch_senses(session: requests.Session, word: str) -> list[dict]:
    """
    Queries the Karp API for the given word and parses the Swedish senses.
    """
    params = {
        "q": f"equals|languages.baseform|{word}",
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
        
        # Check that this matches Swedish baseform
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
        
        # Examples
        examples = []
        for ex in sense.get("examples", []):
            if ex.get("lang") == "swe" and ex.get("text"):
                examples.append(ex["text"])
                
        # Usage
        usage = sense.get("usg", [])
        
        senses.append({
            "id": sense_id,
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
            
            if len(unique_senses) >= 4:
                result[word] = unique_senses
            else:
                print(f"\n[Warning] Word '{word}' has only {len(unique_senses)} valid senses after filtering, skipping...")
        except Exception as e:
            print(f"\n[Error] Failed to process word '{word}': {e}")
        
        # Polite delay between requests
        time.sleep(0.05)
        
    print(f"\nFinished fetching. Writing data to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    duration = time.time() - start_time
    print(f"Successfully compiled {len(result)} words to {OUTPUT_FILE} in {duration:.1f} seconds.")

if __name__ == "__main__":
    main()
