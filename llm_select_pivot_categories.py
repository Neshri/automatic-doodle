#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_select_pivot_categories.py
================================
Feeds a pivot's senses + their scored embedding candidates to an LLM
(gemma4:31b via Ollama) and asks it to make the final judgment call:
which 4 senses make the best puzzle, and which 2 candidates per sense
are the best siblings.

Deliberately restricted to selecting FROM the candidate lists we've
already generated and vetted — not free-generating new Swedish words.
The LLM is being used for judgment/selection (comparing near-duplicates,
avoiding cross-sense collisions, recognizing a sense as a dead end),
which plays to what it's actually reliable at, not for generating
Swedish vocabulary from scratch, which doesn't.

Reuses score_pivots.py's data-loading and candidate-generation directly
rather than duplicating it.

Usage:
  python llm_select_pivot_categories.py --word stoppar
  python llm_select_pivot_categories.py --word stoppar --top-k 20 --model gemma4:31b
  python llm_select_pivot_categories.py --all
  python llm_select_pivot_categories.py --all --output my_results.json --model gemma4:31b
"""

import json
import re
import argparse
import difflib
import requests

from score_pivots import (
    load_embeddings, load_lexicon, get_candidates, sense_spread,
    embed_missing_senses, MULTISENSE_FILE,
)
from generate_multisense_json import get_wiktionary_senses, WIKTIONARY_USER_AGENT


def build_word_inventory(
    lexicon: dict,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, str]]]:
    """
    Returns (inventory_by_pos, all_words_map, definitions_by_pos):
    - inventory_by_pos: dict mapping POS tag ('nn', 'vb', 'av', etc.) to a dict of {baseform_lower: baseform}
    - all_words_map: dict of {baseform_lower: baseform} across all POS
    - definitions_by_pos: dict mapping POS tag to {baseform_lower: definition}, used to backfill
      a definition onto any sibling resolved via the lexicon rather than the candidate list
      (candidates already carry their own definition from get_candidates()).
    """
    inventory_by_pos = {}
    all_words_map = {}
    definitions_by_pos = {}
    for entry in lexicon.values():
        bf = entry.get("baseform")
        pos = entry.get("part_of_speech")
        definition = entry.get("definition", "")
        if bf:
            bf_lower = bf.strip().lower()
            all_words_map[bf_lower] = bf
            if pos:
                inventory_by_pos.setdefault(pos, {})[bf_lower] = bf
                definitions_by_pos.setdefault(pos, {}).setdefault(bf_lower, definition)
    return inventory_by_pos, all_words_map, definitions_by_pos


def verify_and_correct_sibling(
    sib: dict,
    candidate_list: list[dict],
    target_pos: str | None,
    inventory_by_pos: dict[str, dict[str, str]],
    all_words_map: dict[str, str],
    definitions_by_pos: dict[str, dict[str, str]],
    session: requests.Session,
) -> dict:
    """
    Validates and spell-corrects a sibling word returned by the LLM.

    Key principles:
    1. Both 'candidate' and 'suggested' words are validated.
    2. Candidate words are checked against the specific candidate list for the sense.
       If the LLM typo'd a candidate (e.g. 'fantasera' vs candidate 'fantisera'), it is corrected.
    3. Suggested words are checked against the lexicon and Wiktionary ONLY for the target POS.
       Fuzzy matching is strictly POS-restricted, preventing cross-POS mutations (e.g. verb 'förtrösta' -> noun 'förtröstan').
    4. Multi-word phrases with particles ('lita på') are normalized to core baseform ('lita').
    5. Every word that resolves successfully also gets sib['definition'] backfilled --
       from the candidate list, the lexicon, or Wiktionary, whichever path resolved it --
       so the puzzle schema always carries a definition for every word it uses. A word
       that resolves via none of these paths gets 'correction_warning' set instead, and
       carries no definition -- filter_valid_categories() treats that as unusable.
    """
    word = sib.get("word", "").strip()
    if not word:
        return sib
    source = sib.get("source", "candidate")

    cand_map = {c["baseform"].lower(): c["baseform"] for c in candidate_list if c.get("baseform")}
    cand_def_map = {c["baseform"].lower(): c.get("definition", "") for c in candidate_list if c.get("baseform")}

    if source == "candidate":
        if word.lower() in cand_map:
            sib["word"] = cand_map[word.lower()]
            sib["definition"] = cand_def_map.get(word.lower(), "")
            return sib

        # Fuzzy match against the sense's actual candidate list
        cand_matches = difflib.get_close_matches(word.lower(), list(cand_map.keys()), n=1, cutoff=0.82)
        if cand_matches:
            corrected = cand_map[cand_matches[0]]
            print(f"  [spell-fix candidate] '{word}' -> '{corrected}'")
            sib["word"] = corrected
            sib["definition"] = cand_def_map.get(cand_matches[0], "")
            sib["corrected_from"] = word
            return sib

        # Not in candidate list nor a candidate typo -- re-classify as suggested
        source = "suggested"
        sib["source"] = "suggested"

    # --- Suggested word validation ---
    core_word = word.split()[0] if " " in word else word

    if core_word.lower() in cand_map:
        sib["word"] = cand_map[core_word.lower()]
        sib["definition"] = cand_def_map.get(core_word.lower(), "")
        sib["source"] = "candidate"
        return sib

    pos_inv = inventory_by_pos.get(target_pos, {}) if target_pos else all_words_map
    pos_def_map = definitions_by_pos.get(target_pos, {}) if target_pos else {}

    # 1. Exact match in lexicon for target POS
    if core_word.lower() in pos_inv:
        sib["word"] = pos_inv[core_word.lower()]
        sib["definition"] = pos_def_map.get(core_word.lower(), "")
        return sib

    # 2. Check Wiktionary for target POS
    wikt_senses = get_wiktionary_senses(session, core_word)
    matching_wikt = [s for s in wikt_senses if s.get("part_of_speech") == target_pos] if target_pos else wikt_senses
    if matching_wikt:
        sib["word"] = core_word
        sib["definition"] = matching_wikt[0]["definition"]
        sib["wiktionary_definition"] = matching_wikt[0]["definition"]
        return sib

    # 3. Strict POS-restricted fuzzy match in lexicon
    pos_list = list(pos_inv.keys())
    matches = difflib.get_close_matches(core_word.lower(), pos_list, n=1, cutoff=0.87)
    if matches:
        corrected = pos_inv[matches[0]]
        print(f"  [spell-fix suggested POS={target_pos}] '{word}' -> '{corrected}'")
        sib["word"] = corrected
        sib["definition"] = pos_def_map.get(matches[0], "")
        sib["corrected_from"] = word
        return sib

    # Word not found for target POS -- leave word as-is, no definition, flag warning
    sib["correction_warning"] = f"not found in lexicon for POS '{target_pos}'"
    return sib


def build_prompt(word, sense_reports, avg_spread, used_words=None):
    used_words = used_words or set()
    lines = []
    lines.append(f"Pivotord: \"{word}\" (svenska)")
    lines.append(f"Ordet har {len(sense_reports)} betydelser. Ett pussel i Connections-stil behöver högst 4.")
    lines.append(f"Genomsnittligt avstånd mellan betydelserna: avg_pairwise_sim={avg_spread:.3f} "
                 f"(lägre = betydelserna är mer distinkta från varandra, vilket är bra för pusslet).")
    if used_words:
        lines.append("")
        lines.append(f"VIKTIGT: Detta pivotord har redan använts i tidigare pussel i denna omgång. "
                      f"Följande syskonord är REDAN ANVÄNDA och FÅR INTE föreslås igen, varken som "
                      f"\"candidate\" eller \"suggested\": {', '.join(sorted(used_words))}")
    lines.append("")
    lines.append("För varje betydelse nedan: dess definition, samt dess topprankade kandidatord "
                  "(redan filtrerade så att ordets egna andra betydelser är borttagna, rankade efter "
                  "embedding-likhet med betydelsens egen definition — INTE en garanti för att de är bra, bara rankade).")
    lines.append("")

    for i, sr in enumerate(sense_reports, 1):
        lines.append(f"--- BETYDELSE {i}: {sr['id']} [{sr['pos']}] ---")
        lines.append(f"Definition: {sr['definition']}")
        if sr["flags"]:
            lines.append(f"Automatiska flaggor: {', '.join(sr['flags'])}")
        lines.append("Kandidater (poäng, ord, ordklass, definition):")
        for c in sr["candidates"]:
            lines.append(f"  {c['score']:.3f}  {c['baseform']}  [{c['pos']}]  {c['definition']}")
        lines.append("")

    lines.append("""UPPGIFT OCH LINGVISTISKA REGLER FOR PUSSELKVALITET:

1. SEMANTISK SEPARATION (KATEGORIER):
   Kategorierna måste tillhöra helt olika domäner eller beskriva helt olika koncept.
   - Välj ALDRIG två betydelser som bara skiljer sig åt i grammatisk roll (t.ex. transitivt vs. 
     intransitivt), gradskillnad (mild vs. extrem) eller stilnivå för samma grundläggande handling. 
   - Om två betydelser delar samma kärnhandling eller domän, välj endast den starkaste och 
     förkasta den andra i "rejected_senses".

2. MORFOLOGISKT OBEROENDE (SYSKONORD):
   Syskonorden inom en kategori måste vara ortografiskt och etymologiskt oberoende.
   - Orden får INTE dela samma ordstam, ordrot eller vara avledningar/sammensättningar av varandra 
     (t.ex. ett grundord och dess prefix/avledning är inte giltiga syskonord i ett pussel).
   - Syskonorden måste också matcha pivotordets ordklass i den aktuella betydelsen (använd inte 
     substantiv som syskon till ett verbpivot).

3. RIKTNING OCH ANTONYMER:
   Embedding-likhet rankar ofta motsatser högt för att de delar ämne. Kontrollera alltid att 
   kandidatordets faktiska handling rör sig i SAMMA riktning som betydelsens definition (inte motsatt).

4. KVALITET FRAMFÖR KVANTITET:
   Tvinga ALDRIG fram 4 kategorier. Ett pussel med 2 eller 3 klockrena, helt ortogonala kategorier 
   är oändligt mycket bättre än ett pussel med 4 kategorier där någon är sökt, för nära en annan, 
   eller kräver svaga kandidater.

5. KATEGORINAMN (category_label):
   Ge varje kategori ett kort, spelbart namn (2-4 ord) som en spelare skulle se som kategoririd —
   t.ex. "MILITÄR RANG" eller "DEL AV EN BOK".
   - category_label är INTE samma sak som definition (betydelsens ordboksdefinition) och INTE
     samma sak som reasoning (en mening som motiverar varför syskonorden hör ihop). Upprepa inte
     definition eller reasoning ordagrant i category_label.
   - Om definition redan är kort och fungerar fint som ett kategorinamn i sig (t.ex. "dokument"),
     är det okej att category_label liknar den nära — men skriv den ändå som en egen fras, inte en
     kopiering.

6. INGA DUBBLETTER:
   - Samma syskonord får ALDRIG förekomma i mer än en kategori i samma svar — varje ord representeras
     av EN bricka i spelet och kan inte tillhöra två kategorier samtidigt.
   - Om ett ord som skulle passa bra redan är upptaget (antingen av en annan kategori i detta svar,
     eller finns med i listan över REDAN ANVÄNDA SYSKONORD ovan om sådan finns), välj ett annat ord.

INSTRUKTIONER FÖR UTMATNING:
- Välj max 4 betydelser (färre är helt okej).
- För varje vald betydelse: välj EXAKT 2 syskonord. Föredra kandidatlistan ("source": "candidate"). 
  Om kandidatlistan är otillräcklig får du föreslå ord ("source": "suggested"), men använd det återhållsamt.
- Om en betydelse är för lik en annan vald betydelse, eller saknar bra syskonord, placera den i "rejected_senses".

""")
    lines.append("""Du får resonera fritt innan du svarar. Avsluta ditt svar med EXAKT ETT JSON-kodblock i detta
format (och inget annat efter det):
```json
{
  "categories": [
    {"sense_id": "...", "definition": "...", "category_label": "...",
     "siblings": [
        {"word": "...", "source": "candidate"},
        {"word": "...", "source": "suggested"}
     ],
     "root_verification": "Ord 1 rot: [rot], Ord 2 rot: [rot]. Jag bekräftar att de inte delar stam.",
     "reasoning": "en kort mening"}
  ],
  "rejected_senses": [
    {"sense_id": "...", "reason": "en kort mening"}
  ]
}
```""")

    return "\n".join(lines)


def build_feedback_addendum(n_found: int, n_total_senses: int, previous_response: str) -> str:
    """
    Returns a short, neutrally-worded addendum appended to the original prompt
    for a within-attempt retry. Deliberately does NOT mention a required number
    of categories — telling the model it must hit a floor causes it to force
    weak/inaccurate categories to meet the target ("slop"). Instead, the model
    is invited to try a different sense combination and told to apply the same
    quality standard, not a higher quantity target.
    """
    return f"""
---
[Feedback från föregående försök]
Du hittade {n_found} distinkt{'a' if n_found != 1 else ''} kategori{'er' if n_found != 1 else ''} \
från de {n_total_senses} tillgängliga betydelserna.
Undersök om det finns en annan kombination av dessa {n_total_senses} betydelser som ger
fler ortogonala kategorier med lika stark semantisk separation — men bara om de håller
samma kvalitetsstandard som dina regler kräver. Om inte, ange de {n_found} bästa du
hittade och behåll din ursprungliga motivering.

Ditt föregående svar:
{previous_response}
---
"""


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def extract_json(text):
    """
    Pull the JSON object out of a response that may contain reasoning prose
    before it. Prefers a ```json fenced block (what the prompt asks for);
    falls back to brace-matching the first balanced {...} in the text if
    the model didn't fence it properly.
    """
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def call_ollama(prompt, model, temperature, think):
    """
    Streams the response and prints tokens live as they arrive — thinking
    tokens first (if any), then content tokens — so a genuine hang is
    visibly distinguishable from slow-but-working generation. Returns the
    full (content, thinking) strings once the stream completes.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": think,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=120)
    r.raise_for_status()

    full_thinking = []
    full_content = []
    printed_thinking_header = False
    printed_content_header = False

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        message = chunk.get("message", {})
        thinking_piece = message.get("thinking")
        content_piece = message.get("content")

        if thinking_piece:
            if not printed_thinking_header:
                print("\n--- THINKING (live) ---")
                printed_thinking_header = True
            print(thinking_piece, end="", flush=True)
            full_thinking.append(thinking_piece)

        if content_piece:
            if not printed_content_header:
                print("\n--- ANSWER (live) ---")
                printed_content_header = True
            print(content_piece, end="", flush=True)
            full_content.append(content_piece)

        if chunk.get("done"):
            print()
            break

    return "".join(full_content), "".join(full_thinking)


def process_word(word, multisense, matrix, meta, id_to_index, lexicon, args,
                 used_words=None, extra_prompt_suffix=None):
    """
    Build prompt, call LLM, parse result for a single pivot word.
    Returns (result_dict, raw_response, thinking) or raises on hard failure.
    result_dict is None if the word was skipped (too few senses) or parse failed.

    used_words: sibling words already claimed by an earlier puzzle for this
    same pivot (see generate_puzzles_for_word). Filtered out of each sense's
    candidate list before it's even shown to the LLM -- cheaper and more
    reliable than only telling it "don't reuse these" after dangling them
    as top-ranked options.

    extra_prompt_suffix: optional text appended after the main prompt body
    before sending to Ollama -- used by the retry-with-feedback mechanism in
    generate_puzzles_for_word to give the model a second look with context
    from its previous attempt.
    """
    used_words = used_words or set()
    sense_reports = []
    for sense in multisense[word]:
        sid = sense["id"]
        if sid not in id_to_index:
            continue
        pivot_entry = lexicon.get(sid, {})
        pivot_pos = pivot_entry.get("part_of_speech") or sense.get("part_of_speech")
        pivot_baseform = pivot_entry.get("baseform", word)
        candidates = get_candidates(sid, pivot_baseform, matrix, meta, id_to_index, lexicon, args.top_k)
        if used_words:
            candidates = [c for c in candidates if c["baseform"].strip().lower() not in used_words]
        sense_reports.append({
            "id": sid, "pos": pivot_pos, "definition": sense["definition"],
            "flags": [], "candidates": candidates,
        })

    if len(sense_reports) < 4:
        print(f"  Only {len(sense_reports)} senses embedded for '{word}' — need at least 4. Skipping.")
        return None, None, None

    all_sense_ids = [sr["id"] for sr in sense_reports]
    avg_spread, _, _ = sense_spread(all_sense_ids, matrix, id_to_index, close_threshold=0.5)

    prompt = build_prompt(word, sense_reports, avg_spread, used_words=used_words)
    if extra_prompt_suffix:
        prompt = prompt + extra_prompt_suffix

    if args.show_prompt:
        print("=" * 60)
        print(prompt)
        print("=" * 60)

    print(f"  Calling {args.model} (temperature={args.temperature}, think={args.think})...")
    raw_response, thinking = call_ollama(prompt, args.model, args.temperature, args.think)

    if args.show_thinking and thinking:
        print("=" * 60)
        print("THINKING TRACE:")
        print(thinking)
        print("=" * 60)
    elif args.think and not thinking:
        print("  [Note: --think was on, but no thinking trace returned — model may not support it]")

    if args.show_raw:
        print("=" * 60)
        print("RAW RESPONSE:")
        print(raw_response)
        print("=" * 60)

    try:
        json_str = extract_json(raw_response)
        if json_str is None:
            raise json.JSONDecodeError("no JSON object found in response", raw_response, 0)
        result = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"  Could not parse JSON from model response for '{word}'.")
        return None, raw_response, thinking

    # ---- Spell-correct and validate sibling words (POS-aware) ----------
    inventory_by_pos, all_words_map, definitions_by_pos = args._word_inventory_data  # injected into args in main()
    session_for_wikt = requests.Session()
    session_for_wikt.headers.update({"User-Agent": WIKTIONARY_USER_AGENT})
    sense_map = {sr["id"]: sr for sr in sense_reports}

    for cat in result.get("categories", []):
        sid = cat.get("sense_id")
        sr = sense_map.get(sid, {})
        target_pos = sr.get("pos")
        candidate_list = sr.get("candidates", [])

        for sib in cat.get("siblings", []):
            verify_and_correct_sibling(
                sib, candidate_list, target_pos, inventory_by_pos, all_words_map,
                definitions_by_pos, session_for_wikt,
            )
    # --------------------------------------------------------------------

    return result, raw_response, thinking


def filter_valid_categories(categories, used_words):
    """
    Post-hoc safety net -- don't just trust the LLM followed the
    no-duplicate-siblings / don't-reuse-used-words instructions. Walks
    categories in the order the LLM returned them, keeping the first claim
    on any given sibling word and dropping any LATER category that reuses
    a word already claimed -- either by an earlier category in this same
    response, or by an earlier puzzle for this pivot (used_words). A
    colliding category is dropped whole rather than salvaged down to one
    sibling, since a category needs exactly two.

    Also drops any category where a sibling never resolved cleanly
    (correction_warning set by verify_and_correct_sibling -- the word
    couldn't be confirmed to exist for the target POS anywhere) or ended
    up without a definition. Quality gate: the puzzle schema requires a
    definition for every word it uses, so an unverified/undefined sibling
    disqualifies the whole category rather than shipping a hole in it.
    """
    claimed = set(used_words)
    valid = []
    for cat in categories:
        siblings = cat.get("siblings", [])
        words = [s.get("word", "").strip().lower() for s in siblings if s.get("word")]
        if len(words) != 2:
            continue  # malformed -- wrong sibling count
        if len(set(words)) != len(words):
            continue  # category reuses the same word for both its own siblings
        if any(w in claimed for w in words):
            continue  # collides with an earlier category or an earlier puzzle
        if any(s.get("correction_warning") for s in siblings):
            continue  # word never confirmed to exist for the target POS
        if any(not s.get("definition") for s in siblings):
            continue  # schema requires a definition for every word used
        valid.append(cat)
        claimed.update(words)
    return valid


def generate_puzzles_for_word(word, multisense, matrix, meta, id_to_index, lexicon, args):
    """
    Repeatedly calls the LLM for the same pivot, each time telling it which
    sibling words earlier puzzles for this pivot already claimed, so a
    sense-rich pivot (Wiktionary's fallback senses can add several
    near-duplicate senses to one pivot -- e.g. "tro" ended up with three
    different phrasings of "religious belief") can yield SEVERAL distinct
    puzzles instead of forcing everything into one call or throwing the
    near-duplicates away.

    Stops as soon as an attempt can't produce a full 4-category puzzle
    after filtering (build_puzzles.py requires exactly 4 -- a puzzle needs
    4 senses to be valid, no point keeping a partial result), or after
    args.max_puzzles_per_word attempts, whichever comes first.

    On the very first attempt (no puzzles produced yet), if fewer than 4
    valid categories come back, a within-attempt retry fires: the model's
    own response is shown back to it with a neutral nudge to try a different
    sense combination (without revealing the 4-category floor). Controlled
    by args.max_retries_per_attempt (default 1; 0 disables the retry).
    """
    used_words = set()
    puzzles = []

    for attempt in range(1, args.max_puzzles_per_word + 1):
        result, raw_response, thinking = process_word(
            word, multisense, matrix, meta, id_to_index, lexicon, args, used_words=used_words
        )
        if result is None:
            break  # too few senses embedded at all, or JSON parse failure

        categories = filter_valid_categories(result.get("categories", []), used_words)

        # ---- Within-attempt retry (first puzzle only) -------------------
        # Only retry when this is the first puzzle attempt (puzzles list is
        # empty) -- later attempts are expected to have a thinner pool, so
        # retrying them would mostly waste calls.
        max_retries = getattr(args, "max_retries_per_attempt", 1)
        for retry_num in range(1, max_retries + 1):
            if len(categories) >= 4 or len(puzzles) > 0:
                break  # already sufficient, or not the first puzzle attempt
            n_total = len(multisense[word])
            print(f"  '{word}': attempt {attempt} retry {retry_num} — "
                  f"{len(categories)} categories so far, asking for a different sense selection...")
            feedback = build_feedback_addendum(len(categories), n_total, raw_response)
            result, raw_response, thinking = process_word(
                word, multisense, matrix, meta, id_to_index, lexicon, args,
                used_words=used_words, extra_prompt_suffix=feedback,
            )
            if result is None:
                break  # parse failure on retry
            categories = filter_valid_categories(result.get("categories", []), used_words)
        # ---- End retry --------------------------------------------------

        if len(categories) < 4:
            if attempt > 1:
                print(f"  '{word}': attempt {attempt} only yielded {len(categories)} valid categories "
                      f"after collision filtering -- stopping here, keeping {len(puzzles)} puzzle(s).")
            elif max_retries > 0:
                print(f"  '{word}': {len(categories)} valid categories after {max_retries} "
                      f"retr{'y' if max_retries == 1 else 'ies'} — dropping this puzzle.")
            break

        result["categories"] = categories
        puzzles.append(result)

        for cat in categories:
            for sib in cat.get("siblings", []):
                w = sib.get("word", "").strip().lower()
                if w:
                    used_words.add(w)

        print(f"  '{word}': puzzle {attempt} complete ({len(categories)} categories, "
              f"{len(used_words)} sibling words claimed so far).")

    return puzzles


def print_result(word, puzzles):
    """Pretty-print all puzzles generated for a pivot to stdout."""
    if not puzzles:
        print(f"\n########## '{word}': no valid puzzle produced ##########")
        return
    for i, result in enumerate(puzzles, 1):
        categories = result.get("categories", [])
        print(f"\n########## '{word}' — puzzle {i}/{len(puzzles)} ##########")
        print(f"{len(categories)} usable categor{'y' if len(categories) == 1 else 'ies'} found.\n")
        for cat in categories:
            label = cat.get("category_label", "(no label)")
            print(f"[{cat.get('sense_id')}] {label}  —  definition: {cat.get('definition')}")
            for sib in cat.get("siblings", []):
                source = sib.get("source", "candidate")
                definition = sib.get("definition") or "NO DEFINITION"
                corr = sib.get("corrected_from")
                corr_note = f" (corrected from '{corr}')" if corr else ""
                warn = sib.get("correction_warning")
                warn_note = f" ({warn})" if warn else ""

                if source == "candidate":
                    print(f"  {sib.get('word')}{corr_note}  —  {definition}")
                else:
                    print(f"  {sib.get('word')}  <-- SUGGESTED{corr_note}  —  {definition}{warn_note}")
            print(f"  Reasoning: {cat.get('reasoning')}\n")
        if result.get("rejected_senses"):
            print("Rejected senses:")
            for r in result["rejected_senses"]:
                print(f"  {r.get('sense_id')}: {r.get('reason')}")


def main():
    ap = argparse.ArgumentParser()
    word_group = ap.add_mutually_exclusive_group(required=True)
    word_group.add_argument("--word", help="Process a single pivot word")
    word_group.add_argument("--all", action="store_true",
                            help="Process every multisense word and save results to --output")
    ap.add_argument("--output", default="llm_selections.json",
                    help="Output file for --all mode (default: llm_selections.json). "
                         "Existing entries are skipped so the run can be resumed.")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--temperature", type=float, default=0.2,
                     help="Lower = more deterministic. Default lowered from Ollama's default "
                          "after seeing garbled output (typo'd IDs, nonsense tokens) at default temp.")
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=True,
                     help="Enable Ollama's extended-thinking mode if the model supports it (default on). "
                          "Use --no-think to disable.")
    ap.add_argument("--show-prompt", action="store_true", help="Print the full prompt sent to the LLM")
    ap.add_argument("--show-raw", action="store_true", help="Always print the raw LLM response, even on successful parse")
    ap.add_argument("--show-thinking", action="store_true", help="Print the model's thinking trace, if any")
    ap.add_argument("--max-puzzles-per-word", type=int, default=3,
                     help="Cap on how many separate puzzles to try generating from one pivot "
                          "(default 3), so one exceptionally sense-rich pivot doesn't dominate "
                          "the whole set at the expense of variety across different pivots. "
                          "Generation also stops early on its own once an attempt can't reach "
                          "a full 4-category puzzle without reusing an already-claimed sibling.")
    ap.add_argument("--max-retries-per-attempt", type=int, default=1,
                     help="How many within-attempt retries to allow when the first puzzle "
                          "attempt yields fewer than 4 valid categories (default 1). Each "
                          "retry shows the model its own previous response and asks it to "
                          "explore a different sense combination. Use 0 to disable retries "
                          "and restore the original behaviour.")
    args = ap.parse_args()

    with open(MULTISENSE_FILE, "r", encoding="utf-8") as f:
        multisense = json.load(f)

    matrix, meta = load_embeddings()
    lexicon = load_lexicon()
    matrix, meta = embed_missing_senses(multisense, matrix, meta)
    id_to_index = {m["id"]: i for i, m in enumerate(meta)}

    # Build POS-aware word inventory once and attach to args
    args._word_inventory_data = build_word_inventory(lexicon)
    print(f"Word inventory built: {len(args._word_inventory_data[1])} unique baseforms across POS.")

    # ------------------------------------------------------------------ #
    # Single-word mode                                                     #
    # ------------------------------------------------------------------ #
    if args.word:
        if args.word not in multisense:
            print(f"'{args.word}' not in {MULTISENSE_FILE}.")
            return

        puzzles = generate_puzzles_for_word(
            args.word, multisense, matrix, meta, id_to_index, lexicon, args
        )
        print_result(args.word, puzzles)
        return

    # ------------------------------------------------------------------ #
    # Batch mode (--all)                                                  #
    # ------------------------------------------------------------------ #
    # Load existing results so we can resume interrupted runs.
    if args.output and __import__("os").path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            saved = json.load(f)
    else:
        saved = {}

    all_words = sorted(multisense.keys())
    total = len(all_words)
    skipped_already_done = sum(1 for w in all_words if w in saved)
    print(f"Batch mode: {total} words total, {skipped_already_done} already done, "
          f"{total - skipped_already_done} remaining.")
    print(f"Saving results to: {args.output}\n")

    for i, word in enumerate(all_words, 1):
        if word in saved:
            print(f"[{i}/{total}] '{word}' — already done, skipping.")
            continue

        print(f"[{i}/{total}] Processing '{word}'...")
        try:
            puzzles = generate_puzzles_for_word(
                word, multisense, matrix, meta, id_to_index, lexicon, args
            )
        except Exception as exc:
            print(f"  ERROR processing '{word}': {exc}")
            # Record the error so we don't retry the same word on resume
            # (remove this entry manually if you want to retry it).
            saved[word] = {"error": str(exc)}
        else:
            if puzzles:
                print_result(word, puzzles)
                saved[word] = {"puzzles": puzzles}
            else:
                # too-few-senses, parse failure, or first attempt didn't reach
                # 4 valid categories -- record a sentinel so resume skips it
                saved[word] = {"puzzles": []}

        # Write after every candidate so progress is never lost.
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        print(f"  → Saved to {args.output}")

    done = sum(1 for v in saved.values() if v.get("puzzles"))
    total_puzzles = sum(len(v.get("puzzles", [])) for v in saved.values())
    print(f"\nDone. {done}/{total} words produced at least one usable puzzle.")
    print(f"  {total_puzzles} puzzles total across those words.")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()