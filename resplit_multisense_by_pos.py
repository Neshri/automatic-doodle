#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resplit_multisense_by_pos.py
=============================
Fixes an ALREADY-FETCHED multisense_words.json without hitting the Karp API
again — the data itself is fine, it just needs regrouping.

Problem: a word's senses were grouped purely by matching baseform, but same
baseform doesn't mean same word (e.g. noun "stämma" [voice/meeting] and verb
"stämma" [to tune/sue] are unrelated words that happen to share an infinitive
spelling). This splits any pivot whose senses span >1 part_of_speech into
separate entries, one per POS, keyed as "<word>__<pos>".

Words with only one POS present are left with their original, unsuffixed
key — unchanged, so existing single-POS pivots aren't disrupted.

Each resulting POS group still needs >=4 senses to survive — some pivots
that looked like valid 4-sense words under the old grouping will now drop
below threshold once their unrelated homograph's senses are removed. This
is expected: they were never a real 4-sense single word to begin with.

Usage:
  python resplit_multisense_by_pos.py
  python resplit_multisense_by_pos.py --input multisense_words.json --output multisense_words_v2.json
"""

import json
import os
import argparse
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="multisense_words.json")
    ap.add_argument("--output", default="multisense_words_v2.json")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    split_words = []
    dropped = []

    for word, senses in data.items():
        by_pos = defaultdict(list)
        for s in senses:
            by_pos[s["part_of_speech"]].append(s)

        multi_pos = len(by_pos) > 1
        if multi_pos:
            split_words.append((word, list(by_pos.keys())))

        for pos, pos_senses in by_pos.items():
            key = f"{word}__{pos}" if multi_pos else word
            if len(pos_senses) >= 4:
                result[key] = pos_senses
            else:
                dropped.append((key, len(pos_senses), multi_pos))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Read {len(data)} original pivots from {args.input}.")
    print(f"Wrote {len(result)} pivots to {args.output}.")
    print(f"\n{len(split_words)} words had senses spanning >1 POS and were split:")
    for word, poses in split_words:
        print(f"  {word}: {poses}")

    if dropped:
        print(f"\n{len(dropped)} POS-group(s) fell below the 4-sense minimum and were dropped:")
        for key, count, was_split in dropped:
            reason = "after POS split" if was_split else "already too few (shouldn't normally happen)"
            print(f"  {key}: only {count} senses ({reason})")


if __name__ == "__main__":
    main()