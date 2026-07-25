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
    load_embeddings, load_lexicon, get_candidates,
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


def build_sense_category_prompt(word, sense_reports):
    """
    Stage 1 prompt. Deliberately asks for ONE thing only: the best category
    (label + 2 siblings) for EACH sense of the pivot, independently. No
    cross-sense comparison, no rejection decision -- that judgment is
    unreliable when made on abstract dictionary definitions, and it's
    handled later in stage 2 on the concrete categories this stage produces.
    Every sense gets a genuine attempt; a sense with no viable siblings is
    marked unusable with a reason rather than silently skipped, so the
    caller can tell "no good candidates" apart from "not attempted".
    """
    lines = []
    lines.append(f"Pivotord: \"{word}\" (svenska)")
    lines.append(f"Ordet har {len(sense_reports)} betydelser.")
    lines.append("")
    lines.append("För varje betydelse nedan: dess definition, samt dess topprankade kandidatord "
                  "(redan filtrerade så att ordets egna andra betydelser är borttagna, rankade efter "
                  "embedding-likhet med betydelsens egen definition — INTE en garanti för att de är bra, bara rankade).")
    lines.append("")

    for i, sr in enumerate(sense_reports, 1):
        lines.append(f"--- BETYDELSE {i}: {sr['id']} [{sr['pos']}] ---")
        lines.append(f"Definition: {sr['definition']}")
        lines.append("Kandidater (poäng, ord, ordklass, definition):")
        for c in sr["candidates"]:
            lines.append(f"  {c['score']:.3f}  {c['baseform']}  [{c['pos']}]  {c['definition']}")
        lines.append("")

    lines.append("""UPPGIFT OCH LINGVISTISKA REGLER:

Din enda uppgift här är att, FÖR VARJE BETYDELSE OVAN VAR FÖR SIG, hitta de 2 bästa syskonorden
och ett kort spelbart kategorinamn. Du ska INTE jämföra betydelserna mot varandra eller avgöra om
någon är "för lik" en annan -- det görs i ett separat steg senare, med ditt facit som underlag.
Behandla varje betydelse som ett fristående litet problem.

1. MORFOLOGISKT OBEROENDE (SYSKONORD):
   Syskonorden inom en kategori måste vara ortografiskt och etymologiskt oberoende.
   - Orden får INTE dela samma ordstam, ordrot eller vara avledningar/sammensättningar av varandra
     (t.ex. ett grundord och dess prefix/avledning är inte giltiga syskonord i ett pussel).
   - Syskonorden måste också matcha pivotordets ordklass i den aktuella betydelsen (använd inte
     substantiv som syskon till ett verbpivot).

2. RIKTNING OCH ANTONYMER:
   Embedding-likhet rankar ofta motsatser högt för att de delar ämne. Kontrollera alltid att
   kandidatordets faktiska handling rör sig i SAMMA riktning som betydelsens definition (inte motsatt).

3. KATEGORINAMN (category_label):
   Ge varje kategori ett kort, spelbart namn (2-4 ord) som en spelare skulle se som kategoriid —
   t.ex. "MILITÄR RANG" eller "DEL AV EN BOK". Inte samma sak som definition eller reasoning;
   upprepa dem inte ordagrant.

4. SYSKONORD -- KÄLLA:
   Välj EXAKT 2 syskonord, HELST BÅDA från kandidatlistan ("source": "candidate"). Kandidatlistan
   har redan bekräftade ord med definitioner -- ett föreslaget ord ("source": "suggested") måste
   verifieras separat efteråt, och om det misslyckas kasseras HELA kategorin. Föreslå därför bara
   ett ord om kandidatlistan för den betydelsen verkligen saknar något användbart, och välj då ett
   vanligt, etablerat svenskt ord (inte en sällsynt/teknisk term).

5. OM EN BETYDELSE SAKNAR BRA SYSKONORD:
   Tvinga aldrig fram två svaga syskonord. Om varken kandidatlistan eller ett rimligt föreslaget
   ord duger, markera betydelsen som "unusable" med en kort anledning istället för att gissa.

6. INGA DUBBLETTER INOM DETTA SVAR:
   Samma syskonord får inte förekomma i mer än en kategori i hela svaret -- om ett ord passar bra
   för två olika betydelser, använd det bara för en av dem och välj ett alternativ för den andra.

""")
    lines.append("""Du får resonera fritt innan du svarar. Avsluta ditt svar med EXAKT ETT JSON-kodblock i detta
format (och inget annat efter det). En post per betydelse ovan, i samma ordning:
```json
{
  "sense_categories": [
    {"sense_id": "...", "category_label": "...",
     "siblings": [
        {"word": "...", "source": "candidate"},
        {"word": "...", "source": "suggested"}
     ],
     "root_verification": "Ord 1 rot: [rot], Ord 2 rot: [rot]. Jag bekräftar att de inte delar stam.",
     "reasoning": "en kort mening"},
    {"sense_id": "...", "unusable": true, "reason": "en kort mening"}
  ]
}
```""")

    return "\n".join(lines)


def build_selection_prompt(word, categories):
    """
    Stage 2 prompt. Given the CONCRETE categories stage 1 produced (real
    label + real sibling words, not abstract definitions), group them into
    one or more puzzles of up to 4 mutually-distinct categories each. This
    is a comparison over tangible artifacts rather than a judgment call on
    dictionary prose, which is what stage 1 previously conflated it with.

    Sibling words are already globally unique across every category passed
    in here (enforced in code before this prompt is built), so the model
    doesn't need to worry about word collisions -- only about whether two
    categories feel like the same underlying idea to a puzzle solver.
    """
    lines = []
    lines.append(f"Pivotord: \"{word}\" (svenska)")
    lines.append(f"Nedan är {len(categories)} färdiga kategorier, en per betydelse av pivotordet. "
                 f"Varje kategori har redan sina 2 syskonord bestämda.")
    lines.append("")
    lines.append("Din uppgift: gruppera dessa kategorier i ett eller flera pussel om EXAKT 4 kategorier "
                 "vardera, där de 4 kategorierna i samma pussel känns tydligt åtskilda för en spelare -- "
                 "helt olika domäner eller koncept, inte bara olika grammatisk form eller stilnivå av "
                 "samma grundidé. Döm detta på kategorinamnet och syskonorden nedan (det spelaren "
                 "faktiskt ser), inte bara på ordboksdefinitionen.")
    lines.append("")
    lines.append("VIKTIGT -- var inte överförsiktig: två kategorier som råkar komma från samma "
                 "pivotord är INTE automatiskt för lika varandra. Döm varje par på sina egna meriter: "
                 "skulle en spelare som ser de 4 kategorierna sida vid sida uppleva dem som fyra "
                 "olika idéer, eller skulle två av dem kännas som samma sak sagt på två sätt? Bara det "
                 "senare räknas som för likt. Om du är osäker, luta åt att behålla båda snarare än att "
                 "kasta bort en kategori i onödan -- det är kodens jobb att göra en sista kontroll, inte ditt.")
    lines.append("")

    for i, cat in enumerate(categories, 1):
        sib_desc = "; ".join(f"{s['word']} ({s.get('definition', '')})" for s in cat["siblings"])
        lines.append(f"--- KATEGORI {i}: {cat['sense_id']} ---")
        lines.append(f"Namn: {cat['category_label']}")
        lines.append(f"Betydelse (pivotordets definition här): {cat['definition']}")
        lines.append(f"Syskonord: {sib_desc}")
        lines.append("")

    lines.append("""Fler regler:
- Varje kategori får användas i HÖGST ett pussel totalt (inte återanvänd över flera pussel).
- Ett pussel måste ha EXAKT 4 kategorier -- om du bara hittar 2-3 tydligt åtskilda kategorier
  totalt, bilda inget pussel av dem alls hellre än att tvinga ihop ett svagt fjärde val.
- Bilda så många kompletta pussel om 4 som de tillgängliga kategorierna rimligen tillåter --
  men aldrig på bekostnad av att blanda in en kategori som egentligen är för lik en annan i samma pussel.
- Kategorier som inte platsar i något pussel listas i "unused".

Du får resonera fritt innan du svarar. Avsluta ditt svar med EXAKT ETT JSON-kodblock i detta
format (och inget annat efter det):
```json
{
  "puzzles": [
    {"sense_ids": ["...", "...", "...", "..."], "reasoning": "en kort mening"}
  ],
  "unused": [
    {"sense_id": "...", "reason": "en kort mening"}
  ]
}
```""")

    return "\n".join(lines)


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


def build_sense_reports(word, multisense, matrix, meta, id_to_index, lexicon, top_k, used_words=None):
    """
    Deterministic candidate-generation for one pivot's senses -- no LLM call.
    Shared by both generate_sense_categories() callers and any future
    non-LLM inspection of a pivot's candidate pool.
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
        candidates = get_candidates(sid, pivot_baseform, matrix, meta, id_to_index, lexicon, top_k)
        if used_words:
            candidates = [c for c in candidates if c["baseform"].strip().lower() not in used_words]
        sense_reports.append({
            "id": sid, "pos": pivot_pos, "definition": sense["definition"],
            "flags": [], "candidates": candidates,
        })
    return sense_reports


def generate_sense_categories(word, sense_reports, args):
    """
    Stage 1 LLM call: one category (label + siblings) per sense, independently.
    Returns (raw_sense_categories, raw_response, thinking), or (None, raw, thinking)
    on parse failure. raw_sense_categories is the unvalidated list straight from
    the model -- siblings aren't yet resolved/verified, that happens in the caller.
    """
    prompt = build_sense_category_prompt(word, sense_reports)
    if args.show_prompt:
        print("=" * 60)
        print("STAGE 1 PROMPT:")
        print(prompt)
        print("=" * 60)

    print(f"  [stage 1] Calling {args.model} for {len(sense_reports)} senses of '{word}'...")
    raw_response, thinking = call_ollama(prompt, args.model, args.temperature, args.think)

    if args.show_thinking and thinking:
        print("=" * 60)
        print("STAGE 1 THINKING:")
        print(thinking)
        print("=" * 60)

    if args.show_raw:
        print("=" * 60)
        print("STAGE 1 RAW RESPONSE:")
        print(raw_response)
        print("=" * 60)

    try:
        json_str = extract_json(raw_response)
        if json_str is None:
            raise json.JSONDecodeError("no JSON object found in response", raw_response, 0)
        result = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"  [stage 1] Could not parse JSON from model response for '{word}'.")
        return None, raw_response, thinking

    return result.get("sense_categories", []), raw_response, thinking


def resolve_sense_category_pool(word, raw_categories, sense_reports, args, reject_log):
    """
    Verifies/spell-corrects every stage-1 sibling (backfilling definitions),
    logs senses the model marked unusable or skipped entirely, then runs
    filter_valid_categories() ONCE across the whole pool with a single shared
    claimed-word set. Because this happens before any puzzle grouping, every
    surviving category is guaranteed globally word-unique -- stage 2 never
    has to reason about collisions, only about conceptual closeness.

    Returns a dict {sense_id: category} of the categories that survived.
    """
    inventory_by_pos, all_words_map, definitions_by_pos = args._word_inventory_data
    session_for_wikt = requests.Session()
    session_for_wikt.headers.update({"User-Agent": WIKTIONARY_USER_AGENT})
    sense_map = {sr["id"]: sr for sr in sense_reports}

    seen_sids = set()
    candidate_categories = []
    for entry in raw_categories:
        sid = entry.get("sense_id")
        seen_sids.add(sid)
        if entry.get("unusable"):
            reject_log.append({"sense_id": sid, "words": [], "reason": f"stage1_unusable: {entry.get('reason', '')}"})
            continue

        sr = sense_map.get(sid, {})
        target_pos = sr.get("pos")
        candidate_list = sr.get("candidates", [])
        for sib in entry.get("siblings", []):
            verify_and_correct_sibling(
                sib, candidate_list, target_pos, inventory_by_pos, all_words_map,
                definitions_by_pos, session_for_wikt,
            )
        entry["definition"] = sr.get("definition", "")
        candidate_categories.append(entry)

    for sr in sense_reports:
        if sr["id"] not in seen_sids:
            reject_log.append({"sense_id": sr["id"], "words": [], "reason": "sense_missing_from_stage1_response"})

    valid_categories = filter_valid_categories(candidate_categories, used_words=set(), reject_log=reject_log)
    return {cat["sense_id"]: cat for cat in valid_categories}


def select_puzzle_groups(word, category_pool, args):
    """
    Stage 2 LLM call: given the concrete, already-verified categories, group
    them into puzzles of exactly 4 mutually-distinct categories. Retries up
    to args.max_retries_per_attempt times if the model returns zero groups,
    since this is the judgment call most prone to sampling-variance-driven
    over-caution -- and it's cheap to resample, unlike stage 1.

    Returns (groups, unused, reject_log_additions). groups is a list of
    lists of sense_ids (each exactly 4, deduplicated against earlier groups
    in the same response). Malformed groups (wrong size, sense_id reused
    across groups, sense_id not in the pool) are dropped with a logged reason
    rather than silently truncated or crashing.
    """
    categories = list(category_pool.values())
    reject_log = []

    for attempt in range(1, max(1, args.max_retries_per_attempt) + 1):
        prompt = build_selection_prompt(word, categories)
        if args.show_prompt:
            print("=" * 60)
            print(f"STAGE 2 PROMPT (attempt {attempt}):")
            print(prompt)
            print("=" * 60)

        print(f"  [stage 2] Calling {args.model} to group {len(categories)} categories "
              f"into puzzles (attempt {attempt})...")
        raw_response, thinking = call_ollama(prompt, args.model, args.temperature, args.think)

        if args.show_thinking and thinking:
            print("=" * 60)
            print("STAGE 2 THINKING:")
            print(thinking)
            print("=" * 60)
        if args.show_raw:
            print("=" * 60)
            print("STAGE 2 RAW RESPONSE:")
            print(raw_response)
            print("=" * 60)

        try:
            json_str = extract_json(raw_response)
            if json_str is None:
                raise json.JSONDecodeError("no JSON object found in response", raw_response, 0)
            result = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"  [stage 2] Could not parse JSON from model response for '{word}' (attempt {attempt}).")
            continue

        claimed_sids = set()
        groups = []
        for g in result.get("puzzles", []):
            sids = g.get("sense_ids", [])
            if len(sids) != 4:
                reject_log.append({"sense_id": None, "words": sids, "reason": "stage2_group_wrong_size"})
                continue
            if any(s not in category_pool for s in sids):
                reject_log.append({"sense_id": None, "words": sids, "reason": "stage2_group_unknown_sense_id"})
                continue
            if any(s in claimed_sids for s in sids) or len(set(sids)) != 4:
                reject_log.append({"sense_id": None, "words": sids, "reason": "stage2_group_reused_sense_id"})
                continue
            groups.append(sids)
            claimed_sids.update(sids)

        for u in result.get("unused", []):
            reject_log.append({"sense_id": u.get("sense_id"), "words": [],
                                "reason": f"stage2_too_similar: {u.get('reason', '')}"})

        if groups:
            return groups, reject_log

        print(f"  '{word}': stage 2 attempt {attempt} returned zero valid puzzle groups"
              + (", retrying..." if attempt < args.max_retries_per_attempt else "."))

    return [], reject_log


def generate_puzzles_for_word(word, multisense, matrix, meta, id_to_index, lexicon, args):
    """
    Two-stage generation for one pivot:
      1. generate_sense_categories() -- one category per sense, independently.
         No cross-sense judgment, so this call is a narrow, well-defined task
         gemma4 is reliably good at.
      2. resolve_sense_category_pool() -- verify/spell-correct/dedupe, purely
         in code (no LLM), producing a pool of globally word-unique categories.
      3. select_puzzle_groups() -- ask the model which of the CONCRETE, already-
         built categories are distinct enough to co-occur in a puzzle. This is
         the judgment that used to be conflated into step 1 and was causing
         inconsistent over-rejection; splitting it out means it's now made by
         comparing real labels/words instead of abstract definitions, and can
         be cheaply resampled on its own if the model comes back too cautious.

    Puzzles are capped at args.max_puzzles_per_word (a sense-rich pivot can
    otherwise dominate the whole run).

    Returns (puzzles, reject_log) in the same shape as before, so main()'s
    callers don't need to change.
    """
    reject_log = []
    sense_reports = build_sense_reports(word, multisense, matrix, meta, id_to_index, lexicon, args.top_k)

    if len(sense_reports) < 4:
        print(f"  Only {len(sense_reports)} senses embedded for '{word}' — need at least 4. Skipping.")
        return [], reject_log

    raw_categories, raw_response, thinking = generate_sense_categories(word, sense_reports, args)
    if raw_categories is None:
        return [], reject_log  # stage 1 parse failure

    category_pool = resolve_sense_category_pool(word, raw_categories, sense_reports, args, reject_log)
    print(f"  '{word}': {len(category_pool)}/{len(sense_reports)} senses produced a usable category "
          f"after stage 1 verification.")

    if len(category_pool) < 4:
        return [], reject_log  # not enough verified categories to ever form one puzzle

    groups, stage2_rejects = select_puzzle_groups(word, category_pool, args)
    reject_log.extend(stage2_rejects)

    puzzles = []
    for sids in groups[:args.max_puzzles_per_word]:
        categories = [category_pool[sid] for sid in sids]
        puzzles.append({"categories": categories})
        print(f"  '{word}': puzzle {len(puzzles)} complete ({[c['category_label'] for c in categories]}).")

    if not puzzles:
        print(f"  '{word}': stage 1 produced {len(category_pool)} usable categories, "
              f"but stage 2 found no combination of 4 distinct enough to form a puzzle.")

    return puzzles, reject_log


def filter_valid_categories(categories, used_words, reject_log=None):
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

    reject_log: optional list. If given, one entry is appended per dropped
    category -- {"sense_id", "words", "reason"} -- so a batch run over
    hundreds of pivots produces a queryable record of WHY yield was low,
    instead of the reason existing only in scrollback the operator can't
    realistically read for every word.
    """
    claimed = set(used_words)
    valid = []
    for cat in categories:
        siblings = cat.get("siblings", [])
        sid = cat.get("sense_id")
        words = [s.get("word", "").strip().lower() for s in siblings if s.get("word")]

        def _reject(reason):
            if reject_log is not None:
                reject_log.append({"sense_id": sid, "words": words, "reason": reason})

        if len(words) != 2:
            _reject("wrong_sibling_count")
            continue
        if len(set(words)) != len(words):
            _reject("duplicate_word_within_category")
            continue
        if any(w in claimed for w in words):
            _reject("collides_with_already_claimed_word")
            continue
        if any(s.get("correction_warning") for s in siblings):
            _reject("unverified_word")
            continue
        if any(not s.get("definition") for s in siblings):
            _reject("missing_definition")
            continue
        valid.append(cat)
        claimed.update(words)
    return valid


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


def main():
    ap = argparse.ArgumentParser()
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--word", help="Process a single pivot word")
    mode_group.add_argument("--all", action="store_true",
                            help="Process every multisense word and save results to --output")
    ap.add_argument("--output", default="llm_selections.json",
                    help="Output file for --all mode (default: llm_selections.json). "
                         "Existing entries are skipped so the run can be resumed.")
    ap.add_argument("--top-k", type=int, default=40,
                     help="Candidates shown per sense (default 40, up from 20). Widening this is "
                          "the main lever against low yield: every 'suggested' word the model "
                          "reaches for has to independently pass lexicon/Wiktionary/POS "
                          "verification and get a real definition, and that's where most category "
                          "rejections come from. A deeper real candidate pool gives the model less "
                          "reason to invent one.")
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
                     help="Cap on how many puzzles to keep from one pivot (default 3) -- stage 2 "
                          "may propose more if a sense-rich pivot has enough distinct categories; "
                          "this just trims the list so one pivot doesn't dominate the whole set.")
    ap.add_argument("--max-retries-per-attempt", type=int, default=2,
                     help="How many times to re-ask stage 2 (the puzzle-grouping step) if it "
                          "comes back with zero valid groups (default 2). Stage 1 (per-sense "
                          "category generation) is not retried -- it's a narrow, well-defined task "
                          "per sense and doesn't need it. Stage 2's 'are these categories distinct "
                          "enough' judgment is the one prone to sampling-variance over-caution, and "
                          "it's cheap to resample since it doesn't touch the candidate lists again. "
                          "Use 0 to disable retries.")
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

        puzzles, reject_log = generate_puzzles_for_word(
            args.word, multisense, matrix, meta, id_to_index, lexicon, args
        )
        print_result(args.word, puzzles)
        if reject_log:
            print(f"\n{len(reject_log)} categor{'y' if len(reject_log) == 1 else 'ies'} "
                  f"dropped by the quality gate:")
            for r in reject_log:
                print(f"  [{r['sense_id']}] {r['words']} -- {r['reason']}")
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
            puzzles, reject_log = generate_puzzles_for_word(
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
                saved[word] = {"puzzles": puzzles, "reject_log": reject_log}
            else:
                # Sentinel so resume skips it -- but distinguish WHY it's empty:
                # reject_log non-empty means the LLM ran (at least stage 1) and
                # something was rejected along the way (unverified word, stage 2
                # judged it too similar to another sense, etc) -- a candidate for
                # retrying, possibly with different params. An empty reject_log
                # means generation bailed before ever calling the LLM (fewer than
                # 4 senses embedded) -- permanent, retrying won't help. NOTE: both
                # are still skipped on resume; delete the key from the output file
                # manually to force a retry of either kind.
                saved[word] = {"puzzles": [], "reject_log": reject_log}

        # Write after every candidate so progress is never lost.
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        print(f"  → Saved to {args.output}")

    done = sum(1 for v in saved.values() if v.get("puzzles"))
    total_puzzles = sum(len(v.get("puzzles", [])) for v in saved.values())
    needs_retry = sorted(w for w, v in saved.items()
                          if not v.get("puzzles") and v.get("reject_log") and "error" not in v)
    too_few_senses = sorted(w for w, v in saved.items()
                             if not v.get("puzzles") and not v.get("reject_log") and "error" not in v)
    degraded = sorted(w for w, v in saved.items() if v.get("puzzles") and v.get("reject_log"))
    errored = sorted(w for w, v in saved.items() if "error" in v)

    print(f"\nDone. {done}/{total} words produced at least one usable puzzle.")
    print(f"  {total_puzzles} puzzles total across those words.")
    print(f"Results written to {args.output}")
    print(f"\nTriage (no need to eyeball individual entries -- filter the JSON by these):")
    print(f"  {len(needs_retry)} word(s) got LLM output but the quality gate rejected all of it "
          f"(check saved[word]['reject_log']; candidates for --word retry, possibly with a "
          f"higher --top-k so fewer 'suggested' words are needed): {needs_retry[:20]}"
          f"{' ...' if len(needs_retry) > 20 else ''}")
    print(f"  {len(degraded)} word(s) succeeded but dropped >=1 category along the way "
          f"(worth a spot-check via reject_log): {degraded[:20]}{' ...' if len(degraded) > 20 else ''}")
    print(f"  {len(too_few_senses)} word(s) skipped structurally (fewer than 4 senses embedded) "
          f"-- not retryable without more senses.")
    if errored:
        print(f"  {len(errored)} word(s) errored during processing: {errored}")


if __name__ == "__main__":
    main()