#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_multisense_json.py
===========================
Reads words from `multisense_words.txt`, queries the Karp API for Lexin entries,
extracts and formats their Swedish senses, and writes them to a JSON dictionary
where the word is the key and the list of its senses is the value.

PATCHED (lexinID): senses are grouped by lexinID, Lexin's own identifier for
a distinct dictionary entry — NOT by baseform string matching. Verified
directly against raw API output that baseform can be identical for two
unrelated entries (noun "stämma" and verb "stämma" both have
baseform="stämma" but different lexinID: 1144651 vs 1144673), which would
otherwise silently merge them into one pivot.

PATCHED (rawForm re-merge): lexinID groups that share an identical rawForm
(the actual citation spelling) are re-merged before the 4-sense threshold
is applied. lexinID correctly separates genuinely distinct dictionary
entries, but some splits still print IDENTICALLY in-game (e.g. "plan"
splits into an adjective entry + two different-gender noun entries, all
spelled "plan") — checking the 4-sense minimum per lexinID was dropping
each of those individually even though the combined word had enough senses
and would render as one indistinguishable tile. Only entries whose rawForm
actually differs (e.g. noun "stämma" vs verb "stämmer") stay split, since
those can't share one tile.

PATCHED (Wiktionary fallback): if a rawForm group still falls short of the
4-sense minimum on Lexin data alone, additional senses are pulled from
Swedish Wiktionary (via the official MediaWiki API, not scraping) and
appended before giving up on the word. Wiktionary text is CC BY-SA
licensed, which explicitly permits this kind of reuse (unlike SAOL/SO,
which are not open data) — attribution belongs in the project's
README/credits. Lexin is left untouched as the primary source when it
already clears the threshold on its own; Wiktionary only fills the gap.
Each sense is tagged with "source": "lexin" or "source": "wiktionary" so
downstream steps (which need embeddings per sense) know which senses
still need to be embedded before they can flow through score_pivots.py.
"""

import json
import os
import re
import html
import time
from collections import defaultdict
import requests

KARP_API = "https://spraakbanken4.it.gu.se/karp/v7/query/lexin"
WIKTIONARY_API = "https://sv.wiktionary.org/w/api.php"
WIKTIONARY_USER_AGENT = "AutomaticDoodleBot/1.0 (anton/automatic-doodle) python-requests"
INPUT_FILE = "multisense_words.txt"
OUTPUT_FILE = "multisense_words.json"
MIN_SENSES = 4

# Swedish POS labels (Wiktionary section headings) -> SALDO/Lexin-style short
# tags, so Wiktionary-sourced senses carry the same part_of_speech convention
# as Lexin senses elsewhere in the pipeline. Only nn/vb/av/ab/pp have been
# directly confirmed against real Lexin data so far (see score_pivots.py's
# own CLOSED_CLASS_POS comment) -- the rest are reasonable-guess mappings,
# not verified, so double check if one of these classes turns up a lot.
WIKTIONARY_POS_MAP = {
    "substantiv": "nn",
    "verb": "vb",
    "adjektiv": "av",
    "adverb": "ab",
    "preposition": "pp",
    "pronomen": "pn",
    "konjunktion": "kn",
    "interjektion": "in",
}

# Only pull definitions from these Wiktionary headings -- same restriction
# as the original lookup script, so we don't pick up idiom/phrase entries
# etc. as if they were senses of the bare pivot word.
WIKTIONARY_VALID_SECTIONS = set(WIKTIONARY_POS_MAP.keys()) | {
    "räkneord", "artikel", "prefix", "suffix", "partikel", "förkortning",
    "ordstäv", "talesätt", "idiom", "egennamn", "fras", "ordspråk",
}


def get_wiktionary_senses(session: requests.Session, word: str) -> list[dict]:
    """
    Fetches Swedish senses for `word` from Wiktionary via the official
    MediaWiki API (raw wikitext, not scraped HTML). Returns a list of dicts
    matching the same shape as Lexin senses (id/lexin_id/raw_form/
    part_of_speech/definition/phonetic/usage/examples), tagged
    "source": "wiktionary", with synthetic ids like "wiktionary--ord..1".

    Handles the case where Wiktionary splits genuine etymological homographs
    into "== Svenska 1 ==", "== Svenska 2 ==" etc. (as opposed to a single
    "== Svenska ==" section covering all senses of one word, which is what
    most polysemous words -- including domain-spanning ones like "stämma",
    verified directly -- actually use). All matching Svenska section(s) are
    merged into one flat sense list: for our purposes a shared spelling is
    one printable tile regardless of Wiktionary's etymological grouping,
    same principle already applied to Lexin's rawForm merge.
    """
    params = {"action": "parse", "page": word.lower(), "prop": "wikitext", "format": "json"}
    headers = {"User-Agent": WIKTIONARY_USER_AGENT}

    max_retries = 4
    backoff = 5  # seconds, doubles each retry if no Retry-After header is given

    for attempt in range(max_retries + 1):
        try:
            r = session.get(WIKTIONARY_API, params=params, headers=headers, timeout=10)
            if r.status_code == 429:
                if attempt == max_retries:
                    print(f"\n[Warning] Wiktionary still rate-limiting '{word}' after {max_retries} retries, giving up.")
                    return []
                wait = int(r.headers.get("Retry-After", backoff))
                print(f"\n[Info] Wiktionary rate limit hit on '{word}', waiting {wait}s before retry "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                backoff *= 2
                continue
            r.raise_for_status()
            data = r.json()
            break
        except requests.exceptions.HTTPError as e:
            print(f"\n[Warning] Wiktionary fetch failed for '{word}': {e}")
            return []
        except Exception as e:
            print(f"\n[Warning] Wiktionary fetch failed for '{word}': {e}")
            return []

    if "error" in data:
        return []

    wikitext = data["parse"]["wikitext"]["*"]

    # Find every "==Svenska==" / "==Svenska N==" heading -- lenient
    # substring search (no requirement on what immediately precedes/follows
    # the '==' markers, e.g. a language-icon template can sit on the same
    # line), same approach as the original proven-working script. A
    # stricter exact-newline version of this regex was tried first and
    # silently matched nothing on real pages, which is why an earlier
    # version of this function returned 0 senses for every word.
    heading_matches = list(re.finditer(r'==\s*Svenska(?:\s+\d+)?\s*==', wikitext, re.IGNORECASE))
    if not heading_matches:
        return []

    svenska_sections = []
    for hm in heading_matches:
        remainder = wikitext[hm.end():]
        # Bound the section at the next level-2 language heading, same
        # pattern as the original script (uppercase-starting heading on
        # its own line straight after two '=').
        next_lang = re.search(r'\n==\s*[A-Z]', remainder)
        svenska_sections.append(remainder[:next_lang.start()] if next_lang else remainder)

    if not svenska_sections:
        return []

    senses = []
    for sv_text in svenska_sections:
        word_class = None
        for line in sv_text.split('\n'):
            line = line.strip()

            class_match = re.match(r'^={3,}\s*([^=]+?)\s*={3,}$', line)
            if class_match:
                heading = class_match.group(1).strip().lower()
                word_class = heading if heading in WIKTIONARY_VALID_SECTIONS else None
                continue

            if not word_class:
                continue

            def_match = re.match(r'^#+([^*:;].*)', line)
            if not def_match:
                continue

            raw_def = def_match.group(1).strip()
            raw_def = html.unescape(raw_def)
            raw_def = re.sub(r'<!--.*?-->', '', raw_def)
            while re.search(r'\{\{[^{}]*\}\}', raw_def):
                raw_def = re.sub(r'\{\{[^{}]*\}\}', '', raw_def)
            raw_def = re.sub(r'\[\[(?:[^\]|]+\|)?([^\]|]+)\]\]', r'\1', raw_def)
            raw_def = re.sub(r"'{2,}", "", raw_def)
            raw_def = re.sub(r'\[http[^\s]+\s+([^\]]+)\]', r'\1', raw_def)
            raw_def = re.sub(r'\s+', ' ', raw_def).strip()

            if raw_def:
                senses.append({
                    "id": f"wiktionary--{word.lower()}..{len(senses) + 1}",
                    "lexin_id": None,
                    "raw_form": word,
                    "part_of_speech": WIKTIONARY_POS_MAP.get(word_class, word_class),
                    "definition": raw_def,
                    "phonetic": None,
                    "usage": [],
                    "examples": [],
                    "source": "wiktionary",
                })

    return senses


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
            "examples": examples,
            "source": "lexin",
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
    wiktionary_rescued = 0
    
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

            # Group by lexinID first -- Lexin's own ground-truth "distinct
            # dictionary entry" identifier. Two senses can share an
            # identical baseform string while belonging to unrelated
            # entries (verified directly: noun "stämma" and verb "stämma"
            # both have baseform="stämma" but lexinID 1144651 vs 1144673)
            # -- grouping by baseform alone would silently merge them.
            by_lexin_id = defaultdict(list)
            for s in unique_senses:
                by_lexin_id[s["lexin_id"]].append(s)

            # Second pass: re-merge lexinID groups that share an identical
            # rawForm -- see module docstring.
            by_raw_form = defaultdict(list)
            for lexin_id, entry_senses in by_lexin_id.items():
                raw_form = entry_senses[0]["raw_form"]
                by_raw_form[raw_form].extend(entry_senses)

            multi_entry = len(by_raw_form) > 1
            if multi_entry:
                split_count += 1

            for raw_form, entry_senses in by_raw_form.items():
                key = raw_form

                if len(entry_senses) < MIN_SENSES:
                    # Lexin alone isn't enough -- try Wiktionary to fill
                    # the gap before giving up on this word.
                    wiktionary_senses = get_wiktionary_senses(session, raw_form)
                    if wiktionary_senses:
                        # Skip near-duplicate definitions of what Lexin
                        # already gave us, rather than padding the count
                        # with restatements of the same sense.
                        existing_defs = {s["definition"].strip().lower() for s in entry_senses}
                        for ws in wiktionary_senses:
                            if ws["definition"].strip().lower() not in existing_defs:
                                entry_senses.append(ws)
                                existing_defs.add(ws["definition"].strip().lower())
                    time.sleep(0.5)

                    if len(entry_senses) >= MIN_SENSES and any(s["source"] == "wiktionary" for s in entry_senses):
                        wiktionary_rescued += 1

                if len(entry_senses) >= MIN_SENSES:
                    result[key] = entry_senses
                else:
                    if multi_entry:
                        dropped_after_split += 1
                    print(f"\n[Warning] '{key}' has only {len(entry_senses)} valid senses "
                          f"(Lexin + Wiktionary combined) "
                          f"{'(one of multiple distinct spellings under this word) ' if multi_entry else ''}, skipping...")
        except Exception as e:
            print(f"\n[Error] Failed to process word '{word}': {e}")
        
        # Polite delay between requests
        time.sleep(0.05)
        
    print(f"\nFinished fetching. Writing data to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    duration = time.time() - start_time
    print(f"Successfully compiled {len(result)} pivots to {OUTPUT_FILE} in {duration:.1f} seconds.")
    print(f"  {split_count} queried words mapped to >1 distinct rawForm and were split.")
    print(f"  {dropped_after_split} split-off entries fell below the 4-sense minimum even with Wiktionary.")
    print(f"  {wiktionary_rescued} entries were rescued above the 4-sense minimum by Wiktionary.")

if __name__ == "__main__":
    main()