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
import re
import argparse
import requests

from score_pivots import (
    load_embeddings, load_lexicon, get_candidates, sense_spread,
    MULTISENSE_FILE,
)


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
    {"sense_id": "...", "definition": "...",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True)
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

    print(f"\nCalling {args.model} via Ollama (temperature={args.temperature}, think={args.think})...")
    raw_response, thinking = call_ollama(prompt, args.model, args.temperature, args.think)

    if args.show_thinking and thinking:
        print("=" * 60)
        print("THINKING TRACE:")
        print(thinking)
        print("=" * 60)
    elif args.think and not thinking:
        print("[Note: --think was on, but no thinking trace was returned — model may not support it]")

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
        print("Could not find/parse a JSON object in the model's response. Full response:")
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