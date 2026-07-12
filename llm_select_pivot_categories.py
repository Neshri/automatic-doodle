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
"""

import json
import argparse
import requests

from score_pivots import (
    load_embeddings, load_lexicon, get_candidates, sense_spread,
    MULTISENSE_FILE,
)

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


def build_prompt(word, sense_reports, avg_spread):
    lines = []
    lines.append(f"Pivot word: \"{word}\" (Swedish)")
    lines.append(f"It has {len(sense_reports)} senses. A Connections-style puzzle needs exactly 4.")
    lines.append(f"Overall spread between its senses: avg_pairwise_sim={avg_spread:.3f} "
                 f"(lower = senses are more distinct from each other, which is good for the puzzle).")
    lines.append("")
    lines.append("For each sense below: its definition, and its top scored candidate sibling words "
                  "(already filtered to same-headword-excluded, ranked by embedding similarity to the "
                  "sense's own definition — NOT guaranteed to be good, just ranked).")
    lines.append("")

    for i, sr in enumerate(sense_reports, 1):
        lines.append(f"--- SENSE {i}: {sr['id']} [{sr['pos']}] ---")
        lines.append(f"Definition: {sr['definition']}")
        if sr["flags"]:
            lines.append(f"Automated flags: {', '.join(sr['flags'])}")
        lines.append("Candidates (score, word, POS, definition):")
        for c in sr["candidates"]:
            lines.append(f"  {c['score']:.3f}  {c['baseform']}  [{c['pos']}]  {c['definition']}")
        lines.append("")

    lines.append("""TASK:
1. Choose the senses above that make the best puzzle categories — most distinct from
   each other, each with a genuinely good pair of siblings. Up to 4. Fewer than 4 is
   fine and expected if not all senses are strong enough — do NOT force a weak 4th
   pick just to reach the number. A 3-category result you're confident in is better
   than a 4-category result padded with a bad choice.
2. For each chosen sense, pick exactly 2 sibling words. PREFER picking from that
   sense's candidate list above — it's already been vetted for relevance. But if you
   genuinely believe a word NOT in the list fits the sense better (the list can be
   thin or off-target), you may propose it instead. Mark every sibling with
   "source": "candidate" (came from the list) or "source": "suggested" (your own
   addition, not in the list) so suggested words can be spot-checked separately —
   your Swedish vocabulary knowledge is good but not infallible, so be conservative
   about suggesting: only do it when the candidate list is clearly inadequate for
   that sense, not as a default preference over listed candidates.
3. Never pick the same word for two different senses.
4. If a sense has NO good option (candidates are all duplicates, off-topic, or
   function words, AND you can't confidently suggest a better real Swedish word),
   mark that sense as unusable rather than forcing a pick.

Respond ONLY with JSON in exactly this shape, no other text:
{
  "categories": [
    {"sense_id": "...", "definition": "...",
     "siblings": [
        {"word": "...", "source": "candidate"},
        {"word": "...", "source": "suggested"}
     ],
     "reasoning": "one short sentence"}
  ],
  "rejected_senses": [
    {"sense_id": "...", "reason": "one short sentence"}
  ]
}""")

    return "\n".join(lines)


def call_ollama(prompt, model):
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }
    r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--show-prompt", action="store_true", help="Print the full prompt sent to the LLM")
    args = ap.parse_args()

    with open(MULTISENSE_FILE, "r", encoding="utf-8") as f:
        multisense = json.load(f)

    if args.word not in multisense:
        print(f"'{args.word}' not in {MULTISENSE_FILE}.")
        return

    matrix, meta = load_embeddings()
    lexicon = load_lexicon()
    id_to_index = {m["id"]: i for i, m in enumerate(meta)}

    sense_reports = []
    for sense in multisense[args.word]:
        sid = sense["id"]
        if sid not in id_to_index:
            continue
        pivot_entry = lexicon.get(sid, {})
        pivot_pos = pivot_entry.get("part_of_speech")
        pivot_baseform = pivot_entry.get("baseform", args.word)
        candidates = get_candidates(sid, pivot_baseform, matrix, meta, id_to_index, lexicon, args.top_k)
        sense_reports.append({
            "id": sid, "pos": pivot_pos, "definition": sense["definition"],
            "flags": [], "candidates": candidates,
        })

    if len(sense_reports) < 4:
        print(f"Only {len(sense_reports)} senses embedded for '{args.word}' — need at least 4. Aborting.")
        return

    all_sense_ids = [sr["id"] for sr in sense_reports]
    avg_spread, _, _ = sense_spread(all_sense_ids, matrix, id_to_index, close_threshold=0.5)

    prompt = build_prompt(args.word, sense_reports, avg_spread)

    if args.show_prompt:
        print("=" * 60)
        print(prompt)
        print("=" * 60)

    print(f"\nCalling {args.model} via Ollama...")
    raw_response = call_ollama(prompt, args.model)

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError:
        print("Model did not return valid JSON. Raw response:")
        print(raw_response)
        return

    print(f"\n########## LLM selection for '{args.word}' ##########")
    categories = result.get("categories", [])
    print(f"{len(categories)} usable categor{'y' if len(categories) == 1 else 'ies'} found.\n")

    for cat in categories:
        print(f"[{cat.get('sense_id')}] {cat.get('definition')}")
        for sib in cat.get("siblings", []):
            tag = "" if sib.get("source") == "candidate" else "  <-- SUGGESTED, not in candidate list, verify"
            print(f"  {sib.get('word')}{tag}")
        print(f"  Reasoning: {cat.get('reasoning')}\n")

    if result.get("rejected_senses"):
        print("Rejected senses:")
        for r in result["rejected_senses"]:
            print(f"  {r.get('sense_id')}: {r.get('reason')}")


if __name__ == "__main__":
    main()