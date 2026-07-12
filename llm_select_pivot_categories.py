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
    lines.append(f"Pivotord: \"{word}\" (svenska)")
    lines.append(f"Ordet har {len(sense_reports)} betydelser. Ett pussel i Connections-stil behöver högst 4.")
    lines.append(f"Genomsnittligt avstånd mellan betydelserna: avg_pairwise_sim={avg_spread:.3f} "
                 f"(lägre = betydelserna är mer distinkta från varandra, vilket är bra för pusslet).")
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

    lines.append("""EXEMPEL på korrekt resonemang (fiktivt exempel, inte relaterat till ordet ovan):

Betydelse: "får något att sluta brinna" (t.ex. en eld)
Kandidater: 0.800 släcka | 0.750 tända | 0.600 kväva | 0.550 elda

Korrekt val: "släcka" och "kväva" — båda betyder faktiskt att få en eld att sluta brinna.
"tända" är FEL trots hög poäng (0.750) — det betyder motsatsen (att starta en eld), inte att
släcka den. Det rankades högt bara för att det delar ämnet "eld" med definitionen, inte för att
det betyder samma sak. "elda" har samma problem. Detta mönster — hög poäng men fel riktning
eftersom ämnesordet delas men handlingen är motsatt — förekommer ofta. Kontrollera alltid att
ett kandidatords faktiska handling matchar betydelsens definition, inte bara att de delar ämne.

UPPGIFT:
1. Välj de betydelser ovan som ger de bästa pusselkategorierna — mest distinkta från varandra,
   var och en med ett genuint bra par av syskonord. Max 4. Färre än 4 är helt okej och förväntat
   om inte alla betydelser är tillräckligt starka — tvinga INTE fram ett svagt val bara för att
   nå antalet 4. Tre kategorier du är säker på är bättre än fyra där en är dåligt vald.
2. För varje vald betydelse: välj EXAKT 2 syskonord, varken fler eller färre. Föredra att välja
   från kandidatlistan — den är redan granskad för relevans. Men om du är övertygad om att ett
   ord som INTE finns i listan passar bättre (listan kan vara tunn eller missvisande), får du
   föreslå det istället. Märk varje syskonord med "source": "candidate" (från listan) eller
   "source": "suggested" (ditt eget förslag) så att föreslagna ord kan kontrolleras separat —
   var återhållsam med förslag, använd det bara när listan är tydligt otillräcklig.
3. Välj aldrig samma ord för två olika betydelser.
4. Om en betydelse saknar ett bra alternativ, markera den som oanvändbar istället för att tvinga
   fram ett val.

Svara ENDAST med JSON, ingen annan text. Använd EXAKT dessa fältnamn (skrivna på engelska,
som visas här — översätt INTE fältnamnen, bara innehållet):
{
  "categories": [
    {"sense_id": "...", "definition": "...",
     "siblings": [
        {"word": "...", "source": "candidate"},
        {"word": "...", "source": "suggested"}
     ],
     "reasoning": "en kort mening"}
  ],
  "rejected_senses": [
    {"sense_id": "...", "reason": "en kort mening"}
  ]
}""")

    return "\n".join(lines)


def call_ollama(prompt, model, temperature, think=True):
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "think": think,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", ""), data.get("thinking", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--temperature", type=float, default=0.2,
                     help="Lower = more deterministic. Default lowered from Ollama's default "
                          "after seeing garbled output (typo'd IDs, nonsense tokens) at default temp.")
    ap.add_argument("--no-think", action="store_true",
                     help="Disable reasoning/thinking trace in Ollama (default is enabled)")
    ap.add_argument("--show-prompt", action="store_true", help="Print the full prompt sent to the LLM")
    ap.add_argument("--show-raw", action="store_true", help="Always print the raw LLM response, even on successful parse")
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

    print(f"\nCalling {args.model} via Ollama (temperature={args.temperature}, think={not args.no_think})...")
    raw_response, thinking = call_ollama(prompt, args.model, args.temperature, think=not args.no_think)

    if thinking:
        print("=" * 60)
        print("THINKING PROCESS:")
        print(thinking.strip())
        print("=" * 60)

    if args.show_raw:
        print("=" * 60)
        print("RAW RESPONSE:")
        print(raw_response)
        print("=" * 60)

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